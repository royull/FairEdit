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

def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, default='gcn')
        args = parser.parse_known_args()[0]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed = 17

        # set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        dataset = 'german'
        sens_attr = "Gender"  
        sens_idx = 0
        predict_attr = "GoodCustomer"
        label_number = 100
        path_german = "./dataset/german"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number)

        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        
        lr = .01
        weight_decay = 5e-4

        num_class = labels.unique().shape[0]-1
        if args.model == 'gcn':
                model = GCN(nfeat=features.shape[1],
                        nhid=64,
                        nclass=num_class,
                        dropout=0.5)

        elif args.model == 'sage':
                model = SAGE(nfeat=features.shape[1],
                        nhid=64,
                        nclass=num_class,
                        dropout=0.5)
        
        elif args.model == 'appnp':
                model = APPNP(nfeat=features.shape[1],
                        nhid=64,
                        nclass=num_class,
                        K=2, alpha=0.1, dropout=0.5)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model = model.to(device)

        t_total = time.time()
        best_loss = 100
        epochs = 200
        best_acc = 0
        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)

        for epoch in range(epochs):
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

                if loss_val.item() < best_loss:
                        best_loss = loss_val.item()
                        torch.save(model.state_dict(), 'weights/weights_{0}.pt'.format(args.model))
                
main()
