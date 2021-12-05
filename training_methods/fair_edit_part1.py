import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix,from_scipy_sparse_matrix,dropout_adj

def add_drop_edge_random(graph_edge_index,p=0.5,q=0.5):
    """
    Graph_edge_index: Enter Graph edge index
    p: probability of drop edge
    q: probability of add edge
    returns: edge_index
    """
    graph_edge_index, _ = dropout_adj(graph_edge_index, p=p,force_undirected=True)
    B=to_scipy_sparse_matrix(graph_edge_index)
    b=B.toarray()
    n=len(b)
    for i in range(n):
        s=np.random.uniform(0,1,n)
        s=1*(s<q)
        b[:,i]= np.maximum(s,b[:,i])
        b[i,:]= np.maximum(s,b[i,:])
        b[i,i]=0
    temp=coo_matrix(b)
    temp,_=from_scipy_sparse_matrix(temp)

    return temp



