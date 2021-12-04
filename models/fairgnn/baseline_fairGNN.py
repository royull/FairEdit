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


def train(model, x, edge_index, labels, idx_train, sens, optimizer_G, optimizer_A, criterion, args):
    model.train()

    train_g_loss = 0
    train_a_loss = 0

    ### update E, G
    model.adv.requires_grad_(False)
    optimizer_G.zero_grad()

    s = model.estimator(x, edge_index)
    h = model.GNN(x, edge_index)
    y = model.classifier(h)

    s_g = model.adv(h)
    s_score_sigmoid = torch.sigmoid(s.detach())
    s_score = s.detach()
    s_score[idx_train]=sens[idx_train].unsqueeze(1).float()
    y_score = torch.sigmoid(y)
    cov =  torch.abs(torch.mean((s_score_sigmoid[idx_train] - torch.mean(s_score_sigmoid[idx_train])) * (y_score[idx_train] - torch.mean(y_score[idx_train]))))
    
    cls_loss = criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
    adv_loss = criterion(s_g[idx_train], s_score[idx_train])
    G_loss = cls_loss  + args.alpha * cov - args.beta * adv_loss
    G_loss.backward()
    optimizer_G.step()

    ## update Adv
    model.adv.requires_grad_(True)
    optimizer_A.zero_grad()
    s_g = model.adv(h.detach())
    A_loss = criterion(s_g[idx_train], s_score[idx_train])
    A_loss.backward()
    optimizer_A.step()
    return G_loss.detach(), A_loss.detach()


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()
