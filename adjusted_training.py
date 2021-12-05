from os import error
from aif360.sklearn.metrics.metrics import equal_opportunity_difference
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models.gcn import GCN
from models.sage import SAGE
# from models.appnp import APPNP
from training_methods.standard import standard_trainer
from training_methods.brute_force import bf_trainer
<<<<<<< HEAD
from training_methods.fair_edit import fair_edit_trainer
=======
from training_methods.nifty import nifty
>>>>>>> 9c52b6e6b9ee3b62352159f40c15e8bc3c4f4ee8
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert, to_networkx
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
from torch_geometric.utils.homophily import homophily 
from torch_geometric.utils.subgraph import k_hop_subgraph
import matplotlib as mpl
from networkx.algorithms.centrality import closeness_centrality
import matplotlib.pyplot as plt

def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, default='gcn')
        parser.add_argument('--dataset', type=str, default='credit')
        parser.add_argument('--training_method', type=str, default=None)

        args = parser.parse_known_args()[0]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed = 17

        # set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        #### Load Datasets ####
        # Load credit_scoring dataset
        if args.dataset == 'credit':
                sens_attr = "Age"  # column number after feature process is 1
                sens_idx = 1
                predict_attr = 'NoDefaultNextMonth'
                label_number = 6000
                path_credit = "./dataset/credit"
                adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
                                                                                        predict_attr, path=path_credit,
                                                                                        label_number=label_number
                                                                                        )
                norm_features = feature_norm(features)
                norm_features[:, sens_idx] = features[:, sens_idx]
                features = norm_features

        # Load german dataset
        elif args.dataset == 'german':
                sens_attr = "Gender"  # column number after feature process is 0
                sens_idx = 0
                predict_attr = "GoodCustomer"
                label_number = 100
                path_german = "./dataset/german"
                adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(args.dataset, sens_attr,
                                                                                        predict_attr, path=path_german,
                                                                                        label_number=label_number,
                                                                                        )
                edge_index = convert.from_scipy_sparse_matrix(adj)[0]
                
                lr = .01
                weight_decay = 5e-4

        # Load bail dataset
        elif args.dataset == 'bail':
                sens_attr = "WHITE"  # column number after feature process is 0
                sens_idx = 0
                predict_attr = "RECID"
                label_number = 100
                path_bail = "./dataset/bail"
                adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr,
                                                                                        predict_attr, path=path_bail,
                                                                                        label_number=label_number,
                                                                                        )
                norm_features = feature_norm(features)
                norm_features[:, sens_idx] = features[:, sens_idx]
                features = norm_features

        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        num_class = labels.unique().shape[0]-1

        #### Load Models ####
        if args.model == 'gcn':
                model = GCN(nfeat=features.shape[1],
                        nhid=64,
                        nclass=num_class,
                        dropout=0.5)

        elif args.model == 'sage':
                model = SAGE(nfeat=features.shape[1],
                        nhid=64,
                        nclass=num_class,
                        dropout=0.5)
        
        # elif args.model == 'appnp':
        #         model = APPNP(nfeat=features.shape[1],
        #                 nhid=64,
        #                 nclass=num_class,
        #                 K=2, alpha=0.1, dropout=0.5)

        #### Set up training ####
        lr = .01
        weight_decay = 5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model = model.to(device)

        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)

        print("here")
        print(edge_index.shape)
        trainer = None
<<<<<<< HEAD
        if args.training_method == 'standard':   
                trainer = standard_trainer(model=model, dataset=args.dataset, optimizer=optimizer, 
                                                features=features, edge_index=edge_index, 
                                                labels=labels, device=device, train_idx=idx_train, 
                                                val_idx=idx_val)
        elif args.training_method == 'brute':  
                trainer = bf_trainer(model=model, sense_idx = sens_idx, dataset=args.dataset, optimizer=optimizer, 
                                                features=features, edge_index=edge_index, 
                                                labels=labels, device=device, train_idx=idx_train, 
                                                val_idx=idx_val)
        elif args.training_method == 'fairedit':  
                trainer = fair_edit_trainer(model=model, dataset=args.dataset, optimizer=optimizer,
                                                features=features, edge_index=edge_index,
                                                labels=labels, device=device, train_idx=idx_train,
                                                val_idx=idx_val)
=======
        if args.training_method in ['standard','brute','fairedit']:
                if args.training_method == 'standard':   
                        trainer = standard_trainer(model=model, dataset=args.dataset, optimizer=optimizer, 
                                                        features=features, edge_index=edge_index, 
                                                        labels=labels, device=device, train_idx=idx_train, 
                                                        val_idx=idx_val)
                elif args.training_method == 'brute':  
                        trainer = bf_trainer(model=model, sense_idx = sens_idx, dataset=args.dataset, optimizer=optimizer, 
                                                        features=features, edge_index=edge_index, 
                                                        labels=labels, device=device, train_idx=idx_train, 
                                                        val_idx=idx_val)
                        pass
                elif args.training_method == 'fairedit':  
                        trainer = fair_edit_trainer(model=model, dataset=args.dataset, optimizer=optimizer,
                                                        features=features, edge_index=edge_index,
                                                        labels=labels, device=device, train_idx=idx_train,
                                                        val_idx=idx_val)
                trainer.train(epochs=200) # moved up because training epochs are already incorporated into nifty

>>>>>>> 9c52b6e6b9ee3b62352159f40c15e8bc3c4f4ee8
        elif args.training_method == 'nifty':  
                acc, f1s, parity, equality = nifty(features=features,edge_index=edge_index,labels=labels,
                device=device,sens=sens,sens_idx=sens_idx,idx_train=idx_train,idx_test=idx_test,idx_val=idx_val,
                num_class=num_class,lr=lr,weight_decay=weight_decay,model=args.model, hidden=16,proj_hidden=16,
                drop_edge_rate_1=0.001,drop_edge_rate_2=0.001,drop_feature_rate_1=0.1,
                drop_feature_rate_2=0.1,sim_coeff=0.6,epochs=1000)
        else:
                print("Error: Training Method not provided")
                exit(1)



main()
