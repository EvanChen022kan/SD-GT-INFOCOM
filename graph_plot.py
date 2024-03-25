import matplotlib.pyplot as plt
import networkx as nx
import random
import pdb
# Use seed when creating the graph for reproducibility
n = 5
pos = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(n)}

G = nx.random_geometric_graph(5, 3, pos=pos)
# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, "pos")

pdb.set_trace()
# find node near center (0.5,0.5)
dmin = 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = n
        dmin = d

# color by path length from node near center
# p = dict(nx.single_source_shortest_path_length(G, ncenter))

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(
    G,
    pos,
    # nodelist=list(p.keys()),
    node_size=80,
    # node_color=list(p.values()),
    cmap=plt.cm.Reds_r,
)

# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.axis("off")
# plt.show()
plt.grid()
plt.savefig('p1.png', dpi = 300)
