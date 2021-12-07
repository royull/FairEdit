#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import dropout_adj, convert

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='credit',
                    choices=['nba','bail','loan', 'credit', 'german'])



args = parser.parse_known_args()[0]

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
# print(args.dataset)

# Load credit_scoring dataset
for dataset_name in ['german', 'bail', 'credit']:
    args.dataset = dataset_name #This line should be commented
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
    else:
        print('Invalid dataset name!!')
        exit(0)

    print(features.shape)
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    print(edge_index)
    print(labels)


    
