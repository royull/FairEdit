import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix,from_scipy_sparse_matrix,dropout_adj

def add_drop_edge_random(graph_edge_index,features,sens_idx,p=0.1,q=0.1):
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
    for i in range(n):
        #Add edge
        s=np.random.uniform(0,1,n)
        s=1*(s<q)
        b[:,i]= np.maximum(s*sens_matrix[:,i],b[:,i])
        b[i,:]= np.maximum(s*sens_matrix[:,i],b[i,:])
        b[i,i]=0
        #Delete edge
        s = np.random.uniform(0, 1, n)
        s = 1 * (s > p)
        b[:, i] = np.minimum((s + sens_matrix[:, i]), b[:, i])
        b[i, :] = np.minimum((s + sens_matrix[:, i]), b[i, :])
        b[i, i] = 0
    temp=coo_matrix(b)
    temp,_=from_scipy_sparse_matrix(temp)

    return temp
