import dgl
import time
import tqdm
import ipdb
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import dropout_adj, convert

import networkx as nx
from aif360.sklearn.metrics import consistency_score as cs

from models.fairgnn.fgn_utils import *
from models.fairgnn.fairgnn import *
from models.fairgnn.baseline_fairGNN import *


def fgn(args,adj,features,labels, edge_index, idx_train, idx_val, idx_test, sens, device):
    model = FairGNN(nfeat = features.shape[1], args = args).to(device)

    from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score

    # Train model
    t_total = time.time()
    best_result = {}
    best_fair = 100
    G_params = list(model.GNN.parameters()) + list(model.classifier.parameters()) + list(model.estimator.parameters())
    optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
    optimizer_A = torch.optim.Adam(model.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_roc_val = 0
    for epoch in range(args.epochs):
        t = time.time()

        # model.train()
        loss = train(model, features, edge_index, labels, idx_train, sens, optimizer_G, optimizer_A, criterion, args)
        model.eval()
        output, ss, z = model(features, edge_index)
        output_preds = (output.squeeze()>0).type_as(labels)
        ss_preds = (ss.squeeze()>0).type_as(labels)

        # Store accuracy
        acc_val = accuracy(output_preds[idx_val], labels[idx_val]).item()
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
        acc_sens = accuracy(ss_preds[idx_test], sens[idx_test]).item()
        parity_val, equality_val = fair_metric(output_preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].cpu().numpy())

        acc_test = accuracy(output_preds[idx_test], labels[idx_test]).item()
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].cpu().numpy())

        # if epoch % 100 == 0:
        #     print('Epoch: {:04d}'.format(epoch+1), 'acc_val: {:.4f}'.format(acc_val), "roc_val: {:.4f}".format(roc_val), "parity_val: {:.4f}".format(parity_val), "equality: {:.4f}".format(equality_val))  

        if roc_val > best_roc_val:
            best_roc_val = roc_val
            best_result['acc'] = acc_test
            best_result['roc'] = roc_test
            best_result['parity'] = parity
            best_result['equality'] = equality    

            # SaVE models
            torch.save(model.state_dict(), f'./fairgnn_model_{(args.run+1):02d}.pth')
            out_preds = output.squeeze()
            out_preds = (out_preds>0).type_as(labels)

 
    # Load best weights
    model.load_state_dict(torch.load(f'./fairgnn_model_{(args.run+1):02d}.pth'))
    model.eval()
    output, _, _ = model(features, edge_index)
    output_preds = (output.squeeze()>0).type_as(labels)
    ss_preds = (ss.squeeze()>0).type_as(labels)
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output, _, _ = model(noisy_features.to(device), edge_index.to(device))
    noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
    robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
    parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].cpu().numpy())
    f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
    # Add counter factual
    sense_idx = 0
    counter_features = features.clone()
    counter_features[:, sense_idx] = 1 - counter_features[:, sense_idx]
    counter_output,_,_ = model(counter_features.to(device),edge_index.to(device))
    counter_preds = (counter_output.squeeze()>0).type_as(labels)
    counter_fair_score = 1 - (output_preds.eq(counter_preds)[idx_test].sum().item()/idx_test.shape[0])
    
    return auc_roc_test, parity, equality, f1_s


