# python nifty_sota_gnn.py --drop_edge_rate_1 0.001 --drop_edge_rate_2 0.001 --drop_feature_rate_1 0.1 
# --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn 
# --dataset german --sim_coeff 0.6 --seed 1

import time
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models.gcn import *
from models.sage import *
from models.ssf import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee

def ssf_validation(model, x_1, edge_index_1, x_2, edge_index_2, y,idx_val,device,sim_coeff):
    '''
    A supporting function for the main function
    '''

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
    sim_loss = sim_coeff*(l1+l2) ######

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4

# Encoder output
# model = ['gcn','sage']

def nifty(features,edge_index,labels,device,sens,sens_idx,idx_train,idx_test,idx_val,num_class,lr,weight_decay,args):
    '''
    Main Function for NIFTY. Choose 'encode' to be 'gcn' or 'sage' to comply with training.
    Input: listed above. Mostly from args. Some additional been set default value.
    Output: accuracy, f1, parity, counterfactual fairness
    '''
    encoder = Encoder(in_channels=features.shape[1], out_channels=args.hidden, base_model=args.model).to(device)	
    model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff, nclass=num_class).to(device)
    val_edge_index_1 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_1)[0]
    val_edge_index_2 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_2)[0]
    val_x_1 = drop_feature(features.to(device), args.drop_feature_rate_1, sens_idx, sens_flag=False)
    val_x_2 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx)
    par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.fc4.parameters())
    par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
    optimizer_1 = optim.Adam(par_1, lr=lr, weight_decay=weight_decay)
    optimizer_2 = optim.Adam(par_2, lr=lr, weight_decay=weight_decay)
    model = model.to(device)

    # Fairness Training
    t_total = time.time()
    best_loss = 100
    best_acc = 0
    features = features.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)


    for epoch in range(args.epochs+1):
        t = time.time()

        sim_loss = 0
        cl_loss = 0
        rep = 1
        # Lipschtz weight normalization
        for _ in range(rep):
            model.train()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
            x_1 = drop_feature(features, args.drop_feature_rate_2, sens_idx, sens_flag=False)
            x_2 = drop_feature(features, args.drop_feature_rate_2, sens_idx)
            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)

            # projector
            p1 = model.projection(z1)
            p2 = model.projection(z2)

            # predictor
            h1 = model.prediction(p1)
            h2 = model.prediction(p2)

            l1 = model.D(h1[idx_train], p2[idx_train])/2
            l2 = model.D(h2[idx_train], p1[idx_train])/2
            sim_loss += args.sim_coeff*(l1+l2)

        # Fairness Training
        (sim_loss/rep).backward()
        optimizer_1.step()

        # classifier
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        c1 = model.classifier(z1)
        c2 = model.classifier(z2)

        # Binary Cross-Entropy    
        l3 = F.binary_cross_entropy_with_logits(c1[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2
        l4 = F.binary_cross_entropy_with_logits(c2[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2

        cl_loss = (1-args.sim_coeff)*(l3+l4)
        cl_loss.backward()
        optimizer_2.step()
        loss = (sim_loss/rep + cl_loss)

        # Validation
        model.eval()
        val_s_loss, val_c_loss = ssf_validation(model, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2, labels)
        emb = model(val_x_1, val_edge_index_1)
        output = model.predict(emb)
        preds = (output.squeeze()>0).type_as(labels)
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])

        # if epoch % 100 == 0:
        #     print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")

        if (val_c_loss + val_s_loss) < best_loss:
            # print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
            best_loss = val_c_loss + val_s_loss
            torch.save(model.state_dict(), f'weights_ssf_{encoder_model}.pt')

    model.load_state_dict(torch.load(f'weights_ssf_{encoder_model}.pt'))
    model.eval()
    emb = model(features.to(device), edge_index.to(device))
    output = model.predict(emb)
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(model(counter_features.to(device), edge_index.to(device)))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model.predict(model(noisy_features.to(device), edge_index.to(device)))

    # Report
    output_preds = (output.squeeze()>0).type_as(labels)
    counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
    noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
    counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
    robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

    parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
    f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())

    return auc_roc_test, f1_s, parity, counterfactual_fairness