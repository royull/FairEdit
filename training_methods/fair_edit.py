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
from torch_geometric.utils import k_hop_subgraph, to_networkx, sort_edge_index

from ismember import ismember


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
        E, E_p = edge_index.size(1), perturbed_edge_index.size(1)
        
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

        for e in range(0, 2):
            print('gnn_explainer: ' + str(e))
            optimizer.zero_grad()
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
                    labels=None, device=None, train_idx=None, val_idx=None, sens_idx=None, edit_num=20):
        self.model = model
        self.model_name = model.model_name
        self.dataset = dataset
        self.optimizer = optimizer
        self.features = features
        self.edge_index = edge_index
        self.edge_index_orign = copy.deepcopy(self.edge_index)
        self.labels = labels
        self.device = device
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.edit_num = edit_num
        self.perturbed_edge_index = None
        self.sens_idx = sens_idx

    def add_drop_edge_random(self, graph_edge_index, p=0.001,q=0.0001):
        """
        Graph_edge_index: Enter Graph edge index
        p: probability of drop edge
        q: probability of add edge
        returns: edge_index
        """
        # TODO: Can we speed this up at all?
        sens_att=self.features[:, self.sens_idx]
        sens_matrix=torch.outer(sens_att+1,sens_att+1)
        sens_matrix=1*(sens_matrix==2)
        sens_matrix=torch.Tensor.numpy(sens_matrix)
        #graph_edge_index, _ = dropout_adj(graph_edge_index, p=p,force_undirected=True)

        B=to_scipy_sparse_matrix(graph_edge_index)
        b=B.toarray()
        n=len(b)

        #Create Random variable
        s = np.random.uniform(0, 1, (n, n))
        s=s+s.T
        s = 1 * (s < 2*q)
        #Record edges added
        same=((s * sens_matrix)+b)==2
        edges_added =(s * sens_matrix)-1*same
        #Add edge
        b= np.maximum(s*sens_matrix,b)
        b= np.maximum(s*sens_matrix,b)
        #b[i,i]=0

        # Create Random variable
        s1 = np.random.uniform(0, 1, (n,n))
        s1 = s1+s1.T
        s1 = 1 * (s1 > 2*p)
        # Record edges removed
        #same1 = (s1 + sens_matrix + b )== 0
        edges_removed = b*(-np.minimum(s1 + sens_matrix,1)+1)
        #Delete edge
        b = np.minimum((s1 + sens_matrix), b)
        b = np.minimum((s1 + sens_matrix), b)
        #b[i, i] = 0

        temp=coo_matrix(b)
        temp,_=from_scipy_sparse_matrix(temp)
        
        return temp,from_scipy_sparse_matrix(coo_matrix(edges_added))[0],from_scipy_sparse_matrix(coo_matrix(edges_removed))[0]


    def edge_index_to_index1(edge_index,dropped_index):
    """
    Description: edge_list: edge list of input original graph
                 dropped index: edge list dropped from edge_list
                 returns: list of index
    """

    ii,_=ismember(edge_index.T.tolist(), dropped_index.T.tolist(), 'rows')
    return np.where(ii==True)

    def fair_graph_edit(self):
        
        grad_gen = GNNExplainer(self.model)

        # TODO Can we make add/deleted edges indices, not the actual edges?
        self.perturbed_edge_index, add_edges, deleted_edges = self.add_drop_edge_random(self.edge_index)
        print(self.perturbed_edge_index)
        print(add_edges)
        sys.exit()
        edge_mask, perturbed_mask = grad_gen.explain_graph(self.features, self.edge_index, self.perturbed_edge_index)
        
        # TODO get added and deleted to index mask
        added_grads = perturbed_mask[add_edges]
        deleted_grads = edge_mask[deleted_edges]
        print(deleted_grads)
     
        best_add_score = torch.max(added_grads)
        best_add = torch.argmax(added_grads)
        best_add = add_edges[best_add]
        best_delete_score = torch.max(deleted_grads)
        best_delete = torch.argmax(deleted_grads)
        best_delete = deleted_edges[best_delete]
        
        # we want to add edge since better
        if best_add_score < best_delete_score:
            # add both directions since undirected graph
            edge_nodes = self.perturbed_edge_index[:, best_add]
            best_add_comp = torch.tensor([[edge_nodes[1]], [edge_nodes[0]]])
            self.edge_index = torch.cat((self.edge_index, edge_nodes.view(2, 1), best_add_comp), axis=1)
        else: # delete
            edge_nodes = self.edge_index[:, best_delete]
            self.edge_index = torch.cat((self.edge_index[:, :best_delete], self.edge_index[:, best_delete+1:]), axis=1)
            if edge_nodes[0] != edge_nodes[1]:
                vals = (self.edge_index == torch.tensor([[edge_nodes[1]], [edge_nodes[0]]]))
                sum = torch.sum(vals, dim=0)
                col_idx = np.where(sum == 2)[0][0]
                self.edge_index = torch.cat((self.edge_index[:, :col_idx], self.edge_index[:, col_idx+1:]), axis=1)

    def train(self, epochs=200):

        best_loss = 100
        best_acc = 0

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.edge_index)
            print('epoch: ' + str(epoch))
            
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

            if epoch < self.edit_num:
                self.fair_graph_edit()
            
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'fairedit', self.dataset))

