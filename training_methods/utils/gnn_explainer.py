from typing import Optional

from math import sqrt
from inspect import signature

import torch
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
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        self.model = model
        self.lr = lr
        self.__num_hops__ = num_hops
        self.log = log
        self.coeffs.update(kwargs)

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        y = y[subset]

        return x, y, edge_index, mapping, edge_mask

    def __loss__(self, node_idx, log_logits, pred_label):
        
        loss = -log_logits[0, pred_label[0]]
        

        return loss

    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def explain_graph(self, x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, batch=batch, **kwargs)
            log_logits = self.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)

        # TODO Change how set masks is done
        self.__set_masks__(x, edge_index)
        self.to(x.device)
        
        optimizer = torch.optim.Adam(self.edge_mask, lr=self.lr)

        optimizer.zero_grad()
        h = x * self.node_feat_mask.sigmoid()
        out = self.model(x=h, edge_index=edge_index, batch=batch, **kwargs)

        # TODO Update loss 
        log_logits = self.__to_log_prob__(out)
        loss = self.__loss__(-1, log_logits, pred_label)
        loss.backward()
        optimizer.step()

        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return edge_mask



    def __repr__(self):
        return f'{self.__class__.__name__}()'
