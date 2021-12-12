import torch
import numpy as np
import torch.optim as optim
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import APPNP as APPNP_base
from deeprobust.graph import utils
from copy import deepcopy
import torch.nn.functional as F


class APPNP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, K=2, alpha=0.1, dropout=0.5):
        super(APPNP, self).__init__()
        self.model_name = 'appnp'

        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nclass)
        self.prop1 = APPNP_base(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        return x
