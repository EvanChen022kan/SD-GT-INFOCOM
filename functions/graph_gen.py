import torch
import numpy as np
from .gradfn import func
# import pickle
# from .utils import update_fstar, get_fstar, get_mse_fstar
from random import shuffle
import networkx as nx
import random
import pdb


def get_chain_graph(n_agents, batch_size):

    weight = 1/batch_size # equal weights

    G = torch.zeros(n_agents, n_agents)

    Full_subG = weight*torch.ones(batch_size, batch_size)
    batch_num = n_agents//batch_size

    for i in range(batch_num):
        G[i*batch_size: (i+1)*batch_size, i*batch_size: (i+1)*batch_size] = Full_subG
    for i in range(batch_num):
        G[i*batch_size-1, (i+1)*batch_size-1] = 1/3*weight
        G[(i+1)*batch_size-1, i*batch_size-1] = 1/3*weight
        G[(i+1)*batch_size-1, (i+1)*batch_size-1] = 1/3*weight
        if i != batch_num-1:
            G[(i+2)*batch_size-1, (i+1)*batch_size-1] = 1/3*weight
            G[(i+1)*batch_size-1, (i+2)*batch_size-1] = 1/3*weight

    N_in = [[] for i in range(n_agents)]
    for i in range(n_agents):
        for j in range(n_agents):
            if G[i,j] != 0 and i != j:
                N_in[i].append(j)
    # pdb.set_trace()
    return G, N_in

def get_circle_graph(n_agents, N_out_num):
    """
    C is col stochastic -> c[:, i] denotes N_out of i
    R is row stochastic -> R[i, :] denotes N_in of i
    """
    Adj = torch.eye(n_agents)
    Adj = torch.cat([Adj[-1:], Adj[:n_agents-1]])

    for i in range(n_agents):
        nodes = [j for j in range(n_agents) if j != i and j != (i+1) % n_agents]
        shuffle(nodes)
        nodes = nodes[:N_out_num-1]


        for n in nodes:
            Adj[n, i] = 1

    C = (Adj + torch.eye(n_agents)) @ torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=0))
    R = torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=1)) @ (Adj + torch.eye(n_agents))
    
    N_in = [[] for i in range(n_agents)]
    N_out = [[] for i in range(n_agents)]
    for i in range(n_agents):
        for j in range(n_agents):
            if R[i, j] != 0 and i != j:
                N_in[i].append(j)
            if C[j, i] != 0 and i != j:
                N_out[i].append(j)

    return R, C, N_in, N_out


def get_grid_graph(n_agents):
    """
    C is col stochastic -> c[:, i] denotes N_out of i
    R is row stochastic -> R[i, :] denotes N_in of i
    """
    Adj = torch.eye(n_agents, dtype=torch.double)
    # Adj = torch.cat([Adj[-1:], Adj[:n_agents-1]])

    grid_size = 1
    while(True):
        if n_agents <= grid_size**2:
            break
        else:
            grid_size += 1
    # grid_size = grid_size**2
    # pdb.set_trace()

    for i in range(n_agents):
        if i % grid_size >= 1:
            Adj[i, i - 1] = Adj[i - 1, i] = 1
        if i//grid_size >= 1:
            Adj[i, i - grid_size] = Adj[i - grid_size, i] = 1
    # pdb.set_trace()

    '''metropolis'''
    M = Adj.clone() - torch.eye(n_agents, dtype=torch.double)
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and M[i, j] == 1:
                M[i, j] = 1/max(torch.sum(Adj[i]), torch.sum(Adj[j]))
    for i in range(n_agents):
        M[i, i] = 1 - torch.sum(M[i])

    R = C = M

    # C = (Adj + torch.eye(n_agents)) @ torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=0))
    # R = torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=1)) @ (Adj + torch.eye(n_agents))

    N_in = [[] for i in range(n_agents)]
    N_out = [[] for i in range(n_agents)]
    for i in range(n_agents):
        for j in range(n_agents):
            if R[i, j] != 0 and i != j:
                N_in[i].append(j)
            if C[j, i] != 0 and i != j:
                N_out[i].append(j)
    # pdb.set_trace()

    return R, C, N_in, N_out


def get_2star_graph(n_agents):
    """
    C is col stochastic -> c[:, i] denotes N_out of i
    R is row stochastic -> R[i, :] denotes N_in of i
    """
    Adj = torch.eye(n_agents, dtype=torch.double)

    s1 = 0
    s2 = n_agents//2

    for i in range(n_agents):
        for j in range(n_agents):
            if (i == s1 and j < s2) or (j == s1 and i < s2):
                Adj[i, j] = 1
            if (i == s2 and j >= s2) or (j == s2 and i >= s2):
                Adj[i, j] = 1
            if (i == s2 and j == s1) or (j == s2 and i == s1):
                Adj[i, j] = 1

    # pdb.set_trace()

    '''metropolis'''
    M = Adj.clone() - torch.eye(n_agents, dtype=torch.double)
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and M[i, j] == 1:
                M[i, j] = 1/max(torch.sum(Adj[i]), torch.sum(Adj[j]))
    for i in range(n_agents):
        M[i, i] = 1 - torch.sum(M[i])

    R = C = M
    # pdb.set_trace()

    # C = (Adj + torch.eye(n_agents)) @ torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=0))
    # R = torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=1)) @ (Adj + torch.eye(n_agents))

    N_in = [[] for i in range(n_agents)]
    N_out = [[] for i in range(n_agents)]
    # pdb.set_trace()
    for i in range(n_agents):
        for j in range(n_agents):
            if R[i, j] != 0 and i != j:
                N_in[i].append(j)
                # if i == 3:
                #     pdb.set_trace()
            if C[j, i] != 0 and i != j:
                N_out[i].append(j)
    # pdb.set_trace()

    return R, C, N_in, N_out

def get_geo_graph(n_agents):
    """
    C is col stochastic -> c[:, i] denotes N_out of i
    R is row stochastic -> R[i, :] denotes N_in of i
    """
    while(1):
        Adj = torch.eye(n_agents, dtype=torch.double)
        
        pos = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(n_agents)}
        rand_num = 0.5 + 3*random.random()
        G = nx.random_geometric_graph(n_agents, rand_num, pos=pos)
        pos = nx.get_node_attributes(G, "pos")
        edges = [e for e in G.edges]

        for e in edges:
            Adj[e[0], e[1]] = 1
            Adj[e[1], e[0]] = 1


        isconnected = isStronglyConnected(Adj, Adj.shape[0])
        # print("one")
        if isconnected:
            break

    # pdb.set_trace()

    '''metropolis'''
    M = Adj.clone() - torch.eye(n_agents, dtype=torch.double)
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and M[i,j] == 1:
                M[i, j] = 1/max(torch.sum(Adj[i]), torch.sum(Adj[j]))
    for i in range(n_agents):
        M[i, i] = 1 - torch.sum(M[i])

    R = C = M
    # pdb.set_trace()

    # C = (Adj + torch.eye(n_agents)) @ torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=0))
    # R = torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=1)) @ (Adj + torch.eye(n_agents))

    N_in = [[] for i in range(n_agents)]
    N_out = [[] for i in range(n_agents)]
    # pdb.set_trace()
    for i in range(n_agents):
        for j in range(n_agents):
            if R[i, j] != 0 and i != j:
                N_in[i].append(j)
                # if i == 3:
                #     pdb.set_trace()
            if C[j, i] != 0 and i != j:
                N_out[i].append(j)
    # pdb.set_trace()

    return R, C, N_in, N_out


def get_I_graph(n_agents):
    """
    C is col stochastic -> c[:, i] denotes N_out of i
    R is row stochastic -> R[i, :] denotes N_in of i
    """
    Adj = torch.eye(n_agents, dtype=torch.double)


    R = C = Adj
    # pdb.set_trace()

    # C = (Adj + torch.eye(n_agents)) @ torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=0))
    # R = torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=1)) @ (Adj + torch.eye(n_agents))

    N_in = [[] for i in range(n_agents)]
    N_out = [[] for i in range(n_agents)]
    # pdb.set_trace()
    for i in range(n_agents):
        for j in range(n_agents):
            if R[i, j] != 0 and i != j:
                N_in[i].append(j)
                # if i == 3:
                #     pdb.set_trace()
            if C[j, i] != 0 and i != j:
                N_out[i].append(j)
    # pdb.set_trace()

    return R, C, N_in, N_out


def get_ring_graph(n_agents):
    """
    C is col stochastic -> c[:, i] denotes N_out of i
    R is row stochastic -> R[i, :] denotes N_in of i
    """
    Adj = torch.eye(n_agents, dtype=torch.double)
    # Adj = torch.eye(n_agents)
    Adj = torch.cat([Adj[-1:], Adj[:n_agents-1]])

    C = (Adj + torch.eye(n_agents, dtype=torch.double)) @ torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=0))
    R = torch.diag(1/torch.sum(Adj + torch.eye(n_agents, dtype=torch.double), dim=1)) @ (Adj + torch.eye(n_agents, dtype=torch.double))

    # C = (Adj + torch.eye(n_agents)) @ torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=0))
    # R = torch.diag(1/torch.sum(Adj + torch.eye(n_agents), dim=1)) @ (Adj + torch.eye(n_agents))

    N_in = [[] for i in range(n_agents)]
    N_out = [[] for i in range(n_agents)]
    for i in range(n_agents):
        for j in range(n_agents):
            if R[i, j] != 0 and i != j:
                N_in[i].append(j)
            if C[j, i] != 0 and i != j:
                N_out[i].append(j)
    # pdb.set_trace()

    return R, C, N_in, N_out


def get_FC_graph(n_agents):
    """
    C is col stochastic -> c[:, i] denotes N_out of i
    R is row stochastic -> R[i, :] denotes N_in of i
    """
    # pdb.set_trace()
    R = C = torch.ones((n_agents, n_agents), dtype=torch.double)*(1/n_agents)
    # R = C = torch.ones((n_agents, n_agents))*(1/n_agents)


    N_in = [[] for i in range(n_agents)]
    N_out = [[] for i in range(n_agents)]
    for i in range(n_agents):
        for j in range(n_agents):
            if R[i, j] != 0 and i != j:
                N_in[i].append(j)
            if C[j, i] != 0 and i != j:
                N_out[i].append(j)

    return R, C, N_in, N_out


def DFS(graph, v, visited):
    visited[v] = True
    # do for every edge (v, u)
    for u in range(graph.shape[0]):
        if graph[v,u] != 0:
            if not visited[u]:
                DFS(graph, u, visited)


def isStronglyConnected(graph, n):
    # do for every vertex
    for i in range(n):
        # to keep track of whether a vertex is visited or not
        visited = [False] * n
        # start DFS from the first vertex
        DFS(graph, i, visited)
        # If DFS traversal doesn't visit all vertices,
        # then the graph is not strongly connected
        for b in visited:
            if not b:
                return False

    return True
