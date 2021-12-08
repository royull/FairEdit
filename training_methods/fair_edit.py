import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional

from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj
import copy
from math import sqrt
from inspect import signature

from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx


EPS = 1e-15

class GNNExplainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, lr: float = 0.01,
                 num_hops: Optional[int] = None, 
                 log: bool = True, **kwargs):
        super().__init__()
        self.model = model
        self.model_p = copy.deepcopy(model)
        self.lr = lr
        self.__num_hops__ = num_hops
        self.log = log
        self.coeffs.update(kwargs)

    def __set_masks__(self, x, edge_index, perturbed_edge_index, init="normal"):
        (N, F) = x.size()
        E, E_p = edge_index.size(), perturbed_edge_index.size(1)
        
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        print(self.edge_mask.shape)
        self.perturbed_mask = torch.nn.Parameter(torch.randn(E_p) * std)
        print(self.perturbed_mask.shape)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

        for module in self.model_p.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.perturbed_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        for module in self.model_p.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None
        self.perturbed_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __loss__(self, pred, pred_perturb):
        
        loss = torch.norm(pred - pred_perturb, 2)
        
        return loss

    def explain_graph(self, x, edge_index, perturbed_edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.model_p.eval()
        self.__clear_masks__()

        self.__set_masks__(x, edge_index, perturbed_edge_index)
        self.to(x.device)
        
        optimizer = torch.optim.Adam([self.edge_mask, self.perturbed_mask], lr=self.lr)

        for epoch in range(0, 5):
            optimizer.zero_grad()
            print(edge_index.shape)
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            out_p = self.model_p(x=x, edge_index=perturbed_edge_index, **kwargs)
          
            loss = self.__loss__(out, out_p)
            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.detach()
        perturbed_mask = self.perturbed_mask.detach()
        self.__clear_masks__()
        return edge_mask, perturbed_mask
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'


class fair_edit_trainer():
    def __init__(self, model=None, dataset=None, optimizer=None, features=None, edge_index=None, 
                    labels=None, device=None, train_idx=None, val_idx=None, edit_num=20):
        self.model = model
        self.model_name = model.model_name
        self.dataset = dataset
        self.optimizer = optimizer
        self.features = features
        self.edge_index = edge_index
        self.labels = labels
        self.device = device
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.edit_num = edit_num
        self.perturbed_edge_index = None

    def add_drop_edge_random(self, graph_edge_index, p=0.1, q=0.1):
        """
        Graph_edge_index: Enter Graph edge index
        p: probability of drop edge
        q: probability of add edge
        returns: edge_index
        """
        graph_edge_index_d, _ = dropout_adj(graph_edge_index, p=p, force_undirected=True)
        B = to_scipy_sparse_matrix(graph_edge_index_d)
        b = B.toarray()
        n = len(b)
        for i in range(n):
            s = np.random.uniform(0,1,n)
            s = 1*(s<q)
            b[:,i] = np.maximum(s,b[:,i])
            b[i,:] = np.maximum(s,b[i,:])
            b[i,i] = 0
        temp = coo_matrix(b)
        temp, _ = from_scipy_sparse_matrix(temp)

        add_edges = np.array([1, 4, 5])
        deleted_edges = np.array([2, 3])

        return temp, add_edges, deleted_edges

    def fair_graph_edit(self):
        
        grad_gen = GNNExplainer(self.model)

        self.perturbed_edge_index, add_edges, deleted_edges = self.add_drop_edge_random(self.edge_index)
        edge_mask, perturbed_mask = grad_gen.explain_graph(self.features, self.edge_index, self.perturbed_edge_index)
        
        added_grads = perturbed_mask[:, add_edges]
        deleted_grads = edge_mask[:, deleted_edges]
        
        # TODO Figure out if argmin or argmax
        best_add = torch.argmin(added_grads)
        best_delete = torch.argmin(deleted_grads)
        
        # we want to add edge since better
        if best_add < best_delete:
            self.edge_index = torch.cat((self.edge_index, self.perturbed_edge_index[:, best_add]), axis=1)
        else: # delete
            self.edge_index = torch.cat((self.edge_index[:, :best_delete], self.edge_index[:, best_delete+1:]), axis=1)

      

    def train(self, epochs=200):

        best_loss = 100
        best_acc = 0

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.edge_index)
            print(epoch)
            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_train = F.binary_cross_entropy_with_logits(output[self.train_idx], self.labels[self.train_idx].unsqueeze(1).float().to(self.device))

            loss_train.backward()
            self.optimizer.step()
            
            # Evaluate validation set performance separately,
            self.model.eval()
            output = self.model(self.features, self.edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_val = F.binary_cross_entropy_with_logits(output[self.val_idx ], self.labels[self.val_idx ].unsqueeze(1).float().to(self.device))

            while epoch < self.edit_num:
                self.fair_graph_edit()
            
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'fairedit', self.dataset))

