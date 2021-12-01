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

def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, default='gcn')
        parser.add_argument('--dataset', type=str, default='credit')
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
        if args.model == 'fairadj':
                pass

        elif args.model == 'fairwalk':
                pass

        #TODO set up training for these methods
    
main()
