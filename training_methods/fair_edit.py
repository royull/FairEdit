import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional

from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj
import copy
from math import sqrt, floor
from inspect import signature

from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx, sort_edge_index, dense_to_sparse, to_dense_adj

from ismember import ismember


    
EPS = 1e-15

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

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
        self.perturbed_mask = torch.nn.Parameter(torch.randn(E_p) * std)
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
        
        loss = torch.norm(pred - pred_perturb, 1)
        
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

        for e in range(0, 10):
            #print('gnn_explainer: ' + str(e))
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
                    labels=None, device=None, train_idx=None, val_idx=None, sens_idx=None, edit_num=10, sens=None):
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
        self.sens = sens
        counter_features = features.clone()
        counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
        self.counter_features = counter_features
        sens_att = self.features[:, self.sens_idx].int()
        sens_matrix_base = sens_att.reshape(-1, 1) + sens_att
        self.sens_matrix_delete = torch.where(sens_matrix_base != 1, 1, 0).fill_diagonal_(0).int()
        self.sens_matrix_add = torch.where(sens_matrix_base == 1, 1, 0).fill_diagonal_(0).int()
    
    def add_drop_edge_random(self, add_prob=0.001, del_prob=0.01):
        
        N, F = self.features.size() 
        E = self.edge_index.size(1)

        # Get the current graph and filter sensitives based on what is already there
        dense_adj = torch.abs(to_dense_adj(self.edge_index)[0, :, :]).fill_diagonal_(0).int()
        to_delete = torch.logical_and(dense_adj, self.sens_matrix_delete).int()
        to_add = torch.logical_and(dense_adj, self.sens_matrix_add).int()

        # Generate scores 
        scores = torch.Tensor(np.random.uniform(0, 1, (N, N)))

        # DELETE
        masked_scores = scores * to_delete
        masked_scores = torch.triu(masked_scores, diagonal=1)
        num_non_zero = torch.count_nonzero(masked_scores)
        edits_to_make = floor(E * del_prob)
        if num_non_zero < edits_to_make:
            edits_to_make = num_non_zero
        top_delete = torch.topk(masked_scores.flatten(), edits_to_make).indices
        base_end = torch.remainder(top_delete, N)
        base_start = torch.floor_divide(top_delete, N)
        end = torch.cat((base_end, base_start))
        start = torch.cat((base_start, base_end))
        delete_indices = torch.stack([end, start])

        # ADD
        masked_scores = scores * to_add
        masked_scores = torch.triu(masked_scores, diagonal=1)
        num_non_zero = torch.count_nonzero(masked_scores)
        edits_to_make = floor(N**2 * add_prob)
        if num_non_zero < edits_to_make:
            edits_to_make = num_non_zero
        top_adds = torch.topk(masked_scores.flatten(), edits_to_make).indices
        base_end = torch.remainder(top_adds, N)
        base_start = torch.floor_divide(top_adds, N)
        end = torch.cat((base_end, base_start))
        start = torch.cat((base_start, base_end))
        add_indices = torch.stack([end, start])
        
        return delete_indices, add_indices


    def perturb_graph(self, deleted_edges, add_edges):

        # Edges deleted from original edge_index
        delete_indices = []
        self.perturbed_edge_index = copy.deepcopy(self.edge_index)
        for edge in deleted_edges.T:
            vals = (self.edge_index == torch.tensor([[edge[0]], [edge[1]]]))
            sum = torch.sum(vals, dim=0)
            col_idx = np.where(sum == 2)[0][0]
            delete_indices.append(col_idx)

        delete_indices.sort(reverse=True)
        for col_idx in delete_indices:
            self.perturbed_edge_index = torch.cat((self.edge_index[:, :col_idx], self.edge_index[:, col_idx+1:]), axis=1)

        # edges added to perturbed edge_index
        start_edges = self.perturbed_edge_index.shape[1]
        add_indices = [i for i in range(start_edges, start_edges + add_edges.shape[1], 1)]
        self.perturbed_edge_index = torch.cat((self.perturbed_edge_index, add_edges), axis=1)

        return delete_indices, add_indices

    def fair_graph_edit(self):
        
        grad_gen = GNNExplainer(self.model)
    
        # perturb graph (return the ACTUAL edges)
        deleted_edges, added_edges = self.add_drop_edge_random()         
        # get indices of pertubations in edge list (indices in particular edge_lists)             
        del_indices, add_indices = self.perturb_graph(deleted_edges, added_edges) 
        # generate gradients on these perturbations
        edge_mask, perturbed_mask = grad_gen.explain_graph(self.features, self.edge_index, self.perturbed_edge_index)
        added_grads = perturbed_mask[add_indices]
        deleted_grads = edge_mask[del_indices]
     
        # figure out which perturbations were best 
        best_add_score = torch.min(added_grads)
        best_add_idx = torch.argmin(added_grads)
        best_add = added_edges[:, best_add_idx]

        best_delete_score = torch.min(deleted_grads)
        best_delete_idx = torch.argmin(deleted_grads)
        best_delete = deleted_edges[:, best_delete_idx]
        
        # we want to add edge since better
        if best_add_score < best_delete_score:
            # add both directions since undirected graph
            best_add_comp = torch.tensor([[best_add[1]], [best_add[0]]])
            self.edge_index = torch.cat((self.edge_index, best_add.view(2, 1), best_add_comp), axis=1)
        else: # delete
            val_del = (self.edge_index == torch.tensor([[best_delete[1]], [best_delete[0]]]))
            sum_del = torch.sum(val_del, dim=0)
            col_idx_del = np.where(sum_del == 2)[0][0]
            self.edge_index = torch.cat((self.edge_index[:, :col_idx_del], self.edge_index[:, col_idx_del+1:]), axis=1)

            best_delete_comp = torch.tensor([[best_delete[1]], [best_delete[0]]])
            val_del_comp = (self.edge_index == torch.tensor([[best_delete_comp[1]], [best_delete_comp[0]]]))
            sum_del_comp = torch.sum(val_del_comp, dim=0)
            col_idx_del_comp = np.where(sum_del_comp == 2)[0][0]
            self.edge_index = torch.cat((self.edge_index[:, :col_idx_del_comp], self.edge_index[:, col_idx_del_comp+1:]), axis=1)

    def train(self, epochs=200):

        best_loss = 100
        best_acc = 0

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.edge_index)
            #print('epoch: ' + str(epoch))
            
            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_train = F.binary_cross_entropy_with_logits(output[self.train_idx], self.labels[self.train_idx].unsqueeze(1).float().to(self.device))
            f1_train = f1_score(self.labels[self.train_idx].cpu().numpy(), preds[self.train_idx].cpu().numpy())
            loss_train.backward()
            self.optimizer.step()
            
            # Evaluate validation set performance separately,
            self.model.eval()
            output = self.model(self.features, self.edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_val = F.binary_cross_entropy_with_logits(output[self.val_idx ], self.labels[self.val_idx ].unsqueeze(1).float().to(self.device))

            
            f1_val = f1_score(self.labels[self.val_idx ].cpu().numpy(), preds[self.val_idx ].cpu().numpy())
#           Counter factual fairness
            counter_output = self.model(self.counter_features.to(self.device),self.edge_index.to(self.device))
            counter_preds = (counter_output.squeeze()>0).type_as(self.labels)
            fair_score = 1 - (preds.eq(counter_preds)[self.val_idx].sum().item()/self.val_idx.shape[0])
#           Robustness    
            noisy_features = self.features.clone() + torch.ones(self.features.shape).normal_(0, 1).to(self.device)
            noisy_output = self.model(noisy_features, self.edge_index)
            noisy_output_preds = (noisy_output.squeeze()>0).type_as(self.labels)
            robustness_score = 1 - (preds.eq(noisy_output_preds)[self.val_idx].sum().item()/self.val_idx.shape[0])
            parity, equality = fair_metric(preds[self.val_idx].cpu().numpy(), self.labels[self.val_idx].cpu().numpy(), self.sens[self.val_idx].numpy())

            if epoch < self.edit_num:
                print(epoch)
                self.fair_graph_edit()
            

            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'fairedit', self.dataset))
        print("== f1: {} fair: {} robust: {}, parity:{} equility: {}".format(f1_val,fair_score,robustness_score,parity,equality))
        return None, f1_val, fair_score, robustness_score, parity, equality


