import argparse
import torch.nn as nn
import torch
import numpy as np
import random



class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default='Example', help="Decide your networks name! :)")
        self.parser.add_argument('--dim', type=int, default='300')
        self.parser.add_argument('--gpu_id', type=int, default=2)
        self.parser.add_argument('--batch', type=int, default=8)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--beta', type=float, default=1e-1)



        # self.parser.add_argument('--beta', type=int, default=1)
        self.parser.add_argument('--sample_num', type=int, default=1)


        self.parser.add_argument('--n_agents', type=int, default='30')
        self.parser.add_argument('--n_cluster', type=int, default='6')
        self.parser.add_argument('--K_val', type=int, default='5')


        self.parser.add_argument('--kappa', type=float, default=1e-4)
        self.parser.add_argument('--GTarg', type=int, default=1)

        self.parser.add_argument('--omega', type=float, default=0.1)
        self.parser.add_argument('--loss', type=str, default='mse', choices=["mse", "log"])
        self.parser.add_argument('--data_source', type=str, default='generated', choices=["w8a", "generated"])

        self.parser.add_argument('--use_ring', action="store_true")
        self.parser.add_argument('--use_grid', action="store_true")
        self.parser.add_argument('--use_2star', action="store_true")
        self.parser.add_argument('--use_I', action="store_true")
        self.parser.add_argument('--random', action="store_true")
        self.parser.add_argument('--delta', type=float, default=0)

        
        self.parser.add_argument('--control', action="store_true")
        self.parser.add_argument('--new_control', action="store_true")
        self.parser.add_argument('--update_rounds', type=int, default=20)
        



        self.parser.add_argument('--dataset', type=str, default='MNIST', choices=["MNIST", "CIFAR10", 'CIFAR100'])
        self.parser.add_argument('--model', type=str, default='FNN', choices=["FNN", "CNN", "CNN2", "TOMCNN"])





        self.parser.add_argument('--scatter', action="store_true")

        self.parser.add_argument('--p_iter', type=int, default='50')




        # self.parser.add_argument('--f_star', type=float, default=0.16601014612859405) # for 1e-4
        # self.parser.add_argument('--f_star', type=float, default=0.2365823289406) # for 1e-3
        # self.parser.add_argument('--f_star', type=float, default=0.1162383787517) # for 1e-6
        # self.parser.add_argument('--f_star', type=float, default=0.1272720938340)  # for 1e-4 new L
        # self.parser.add_argument('--f_star', type=float, default=0.12727255776359632)  # for 1e-4 new L
        # self.parser.add_argument('--f_star', type=float, default=0.5764672258805)  # for 1e-4 new L

        # 0.1272726118340



        
        
        self.parser.add_argument('--result_path', type=str, default='results')

        self.parser.add_argument('--c_rounds', type=int, default=1000)
        self.parser.add_argument('--p_inv', type=float, default=1000)
        self.parser.add_argument('--LR_c', type=float, default=1)

        self.parser.add_argument('--test_prox', action="store_true")
        self.parser.add_argument('--test_ours', action="store_true")
        self.parser.add_argument('--test_both', action="store_true")
        self.parser.add_argument('--test_LR', action="store_true")
        self.parser.add_argument('--test_beta', action="store_true")
        self.parser.add_argument('--test_diff', action="store_true")
        self.parser.add_argument('--test_asy', action="store_true")
        self.parser.add_argument('--test_sync', action="store_true")
        self.parser.add_argument('--test_cen', action="store_true")
        self.parser.add_argument('--comp_prox', action="store_true")
        self.parser.add_argument('--sync_only', action="store_true")

        self.parser.add_argument('--setting1', action="store_true")
        self.parser.add_argument('--setting2', action="store_true")
        self.parser.add_argument('--setting3', action="store_true")


        self.parser.add_argument('--config_id', type=int, default=1)

        




        
        # self.parser.add_argument('--detail_search', action="store_true")










        # self.parser.add_argument('--i_rounds', type=int, default=1000)

    def get_options(self):
        return self.parser.parse_args()