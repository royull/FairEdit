import dgl
import ipdb
import time
import argparse
import numpy as np
from torch_geometric.utils import convert

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models.fairgnn.fairgnn_wrapped import fgn

def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, default='gcn')
        parser.add_argument('--training', type=str, default='fairgnn')
        parser.add_argument('--dataset', type=str, default='credit')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units of the sensitive attribute estimator')
        parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')

        ### arguments for FairGNN
        parser.add_argument('--alpha', type=float, default=4,
                        help='The hyperparameter of alpha')
        parser.add_argument('--beta', type=float, default=0.01,
                        help='The hyperparameter of beta')
        parser.add_argument('--num-hidden', type=int, default=32,
                        help='Number of hidden units of classifier.')
        parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
        parser.add_argument('--run', type=int, default=0,
                        help="kth run of the model")

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

        
        
        lr = .01
        weight_decay = 5e-4

        num_class = labels.unique().shape[0]-1
        
        if args.training == 'fairgnn':
                f1s, parity, equality, counter, robust = fgn(args,adj,features,labels, edge_index, idx_train, idx_val, idx_test, sens, device,sens_idx)
                print(f1s,counter,robust,parity,equality)
        elif args.model == 'fairwalk':
                pass
    
main()
