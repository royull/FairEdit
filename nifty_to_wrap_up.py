#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models.gcn import *
from models.sage import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def ssf_validation(model, x_1, edge_index_1, x_2, edge_index_2, y):
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1[idx_val], p2[idx_val])/2
    l2 = model.D(h2[idx_val], p1[idx_val])/2
    sim_loss = args.sim_coeff*(l1+l2)

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--proj_hidden', type=int, default=16,
                    help='Number of hidden units in the projection layer of encoder.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--drop_edge_rate_1', type=float, default=0.1,
                    help='drop edge for first augmented graph')
parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,
                    help='drop edge for second augmented graph')
parser.add_argument('--drop_feature_rate_1', type=float, default=0.1,
                    help='drop feature for first augmented graph')
parser.add_argument('--drop_feature_rate_2', type=float, default=0.1,
                    help='drop feature for second augmented graph')
parser.add_argument('--sim_coeff', type=float, default=0.5,
                    help='regularization coeff for the self-supervised task')
parser.add_argument('--dataset', type=str, default='loan',
                    choices=['nba','bail','loan', 'credit', 'german'])
parser.add_argument("--num_heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn'])
parser.add_argument('--encoder', type=str, default='gcn')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
# print(args.dataset)

# Load credit_scoring dataset
if args.dataset == 'credit':
	sens_attr = "Age"  # column number after feature process is 1
	sens_idx = 1
	predict_attr = 'NoDefaultNextMonth'
	label_number = 6000
	path_credit = "./dataset/credit"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_credit,
	                                                                        label_number=label_number
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features

# Load german dataset
elif args.dataset == 'german':
	sens_attr = "Gender"  # column number after feature process is 0
	sens_idx = 0
	predict_attr = "GoodCustomer"
	label_number = 100
	path_german = "./dataset/german"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_german,
	                                                                        label_number=label_number,
	                                                                        )
# Load bail dataset
elif args.dataset == 'bail':
	sens_attr = "WHITE"  # column number after feature process is 0
	sens_idx = 0
	predict_attr = "RECID"
	label_number = 100
	path_bail = "./dataset/bail"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr, 
																			predict_attr, path=path_bail,
	                                                                        label_number=label_number,
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features
else:
	print('Invalid dataset name!!')
	exit(0)

edge_index = convert.from_scipy_sparse_matrix(adj)[0]

# Model and optimizer
num_class = labels.unique().shape[0]-1
if args.model == 'gcn':
	model = GCN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

elif args.model == 'sage':
	model = SAGE(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

# Train model
t_total = time.time()
best_loss = 100
best_acc = 0
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)

for epoch in range(args.epochs+1):
    t = time.time()

    if args.model in ['gcn', 'sage']:
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)

        # Binary Cross-Entropy  
        preds = (output.squeeze()>0).type_as(labels)
        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        model.eval()
        output = model(features, edge_index)

        # Binary Cross-Entropy
        preds = (output.squeeze()>0).type_as(labels)
        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))

        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())

        # if epoch % 100 == 0:
        #     print(f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            torch.save(model.state_dict(), 'weights_vanilla.pt')


if args.model in ['gcn', 'sage']:
    model.load_state_dict(torch.load('weights_vanilla.pt'))
    model.eval()
    output = model(features.to(device), edge_index.to(device))
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model(counter_features.to(device), edge_index.to(device))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model(noisy_features.to(device), edge_index.to(device))

# Report
output_preds = (output.squeeze()>0).type_as(labels)
counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())

# print report
print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
print(f'Parity: {parity} | Equality: {equality}')
print(f'F1-score: {f1_s}')
print(f'CounterFactual Fairness: {counterfactual_fairness}')
print(f'Robustness Score: {robustness_score}')
