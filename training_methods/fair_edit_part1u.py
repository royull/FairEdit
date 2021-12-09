import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix,from_scipy_sparse_matrix,dropout_adj
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from ismember import ismember


def add_drop_edge_random(graph_edge_index,features,sens_idx,p=0.5,q=0.5):
    """
    Graph_edge_index: Enter Graph edge index
    p: probability of drop edge
    q: probability of add edge
    returns: edge_index
    """
    sens_att=features[:,sens_idx]
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
    #b[i, i] = 0

    temp=coo_matrix(b)
    temp,_=from_scipy_sparse_matrix(temp)

    return temp,from_scipy_sparse_matrix(coo_matrix(edges_added))[0],from_scipy_sparse_matrix(coo_matrix(edges_removed))[0]

def edge_index_to_index1(edge_index,dropped_index):
    
    Description: edge_list: edge list of input original graph
                 dropped index: edge list dropped from edge_list
                 returns: list of index
    
    ii,_=ismember(edge_index.T.tolist(), dropped_index.T.tolist(), 'rows')
    return np.where(ii==True)

def Graph_sdd_drop_sanity_check():
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 0, 5, 1, 2, 4, 4, 5, 0, 5],
                               [1, 0, 2, 1, 0, 3, 1, 5, 4, 2, 5, 4, 5, 0]], dtype=torch.long)
    x = torch.tensor([[1], [0], [1], [1], [1], [0]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    plt.figure(1)
    plt.clf()
    nx.draw(to_networkx(data, to_undirected=True), with_labels=True)

    e, a, d = add_drop_edge_random(edge_index, x, 0, p=0.5, q=0.5)
    data_e = Data(x=x, edge_index=e)

    plt.figure(2)
    plt.clf()
    nx.draw(to_networkx(data_e, to_undirected=True), with_labels=True)
    print("Add\n", a)
    print("Drop\n", d)
