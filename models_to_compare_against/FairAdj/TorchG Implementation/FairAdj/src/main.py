# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import scipy.sparse as sp

import torch
from torch import optim
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch_sparse


from args import parse_args
from utils import fix_seed, find_link
from dataloader import get_dataset
from model.utils import preprocess_graph, project
from model.optimizer import loss_function
from model.G_VGAE import VGAE
from eval import fair_link_eval


def main(args):
    # Data preparation
    G, adj, features, sensitive, test_edges_true, test_edges_false = get_dataset(args.dataset, args.scale,
                                                                                 args.test_ratio)
    n_nodes, feat_dim = features.shape
    features = torch.from_numpy(features).float().to(args.device)
    sensitive_save = sensitive.copy()
    adj = torch.FloatTensor(adj.toarray()).to(args.device)
    # adj is adjacent Matrix, type: torch.tensor
    # features is node's feature, type: torch.tensor
    # sensitive is the sensitivity type of each node, type: np.ndarray
    # test_edges_xxx is only for validation, type: list
    # TODO: @ROY: align the dataloader with it
    sp_sparse_adj = sp.coo_matrix(adj.numpy())
    # To utilize some of the helper functions in original implementation, make a copy of adj in scipy.sparse.coo_matrix format
    adj_norm = preprocess_graph(sp_sparse_adj).to(args.device).to_dense()
    # Note: This is the final representation used during comp
    adj_norm = SparseTensor.from_dense(adj_norm)

    # Type: torch.sparse
    sp_sparse_adj = sp.coo_matrix(
        sp_sparse_adj + sp.eye(sp_sparse_adj.shape[0]))
    adj_label = torch.Tensor(sp_sparse_adj.toarray()).to(args.device).float()

    intra_pos, inter_pos, intra_link_pos, inter_link_pos = find_link(
        sp_sparse_adj, sensitive)

    pos_weight = float(
        sp_sparse_adj.shape[0] * sp_sparse_adj.shape[0] - sp_sparse_adj.sum()) / sp_sparse_adj.sum()
    pos_weight = torch.Tensor([pos_weight]).to(args.device)
    norm = sp_sparse_adj.shape[0] * sp_sparse_adj.shape[0] / \
        float(
            (sp_sparse_adj.shape[0] * sp_sparse_adj.shape[0] - sp_sparse_adj.sum()) * 2)

    # Initialization
    # model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(args.device)
    model = VGAE(feat_dim, args.hidden1, args.hidden2).to(args.device)
    optimizer = optim.Adam(model.get_parameters(), lr=args.lr)

    # Training
    model.train()
    for i in range(args.outer_epochs):
        for epoch in range(args.T1):
            optimizer.zero_grad()

            recovered, z, mu, logvar = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label, mu=mu, logvar=logvar, n_nodes=n_nodes, norm=norm,
                                 pos_weight=pos_weight)

            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            print("Epoch in T1: [{:d}/{:d}];".format((epoch + 1),
                  args.T1), "Loss: {:.3f};".format(cur_loss))

        for epoch in range(args.T2):
            adj_norm.storage._value.requires_grad = True
            recovered = model(features, adj_norm)[0]

            if args.eq:
                intra_score = recovered[intra_link_pos[:,
                                                       0], intra_link_pos[:, 1]].mean()
                inter_score = recovered[inter_link_pos[:,
                                                       0], inter_link_pos[:, 1]].mean()
            else:
                intra_score = recovered[intra_pos[:, 0],
                                        intra_pos[:, 1]].mean()
                inter_score = recovered[inter_pos[:, 0],
                                        inter_pos[:, 1]].mean()

            loss = F.mse_loss(intra_score, inter_score)
            loss.backward()
            cur_loss = loss.item()

            print("Epoch in T2: [{:d}/{:d}];".format(epoch +
                  1, args.T2), "Loss: {:.5f};".format(cur_loss))
            # grad_values = adj_norm.storage._value.grad
            adj_norm_grad_dense = SparseTensor(col = adj_norm.storage._col, row = adj_norm.storage._row, value=adj_norm.storage._value.grad).to_dense()


            # adj_norm = adj_norm.add(adj_norm.grad.mul(-args.eta)).detach()
            adj_norm = (adj_norm.to_dense() + adj_norm_grad_dense.mul(-args.eta)).detach()

            # adj_norm = adj_norm.add(adj_norm_grad_sp.mul(-args.eta)).detach()
            # adj_norm = adj_norm.to_dense()

            for i in range(adj_norm.shape[0]):
                adj_norm[i] = project(adj_norm[i])

            # adj_norm = adj_norm.to_sparse()
            adj_norm = SparseTensor.from_dense(adj_norm)

    # Evaluation
    model.eval()
    with torch.no_grad():
        z = model(features, adj_norm)[1]
    hidden_emb = z.data.cpu().numpy()

    std = fair_link_eval(hidden_emb, sensitive_save,
                         test_edges_true, test_edges_false)
    col = ["auc", "ap", "dp", "true", "false", "fnr", "tnr"]
    print("Result below ------")
    for term, val in zip(col, std):
        print(term, ":", val)

    return


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device(args.device)
    fix_seed(args.seed)
    main(args)
