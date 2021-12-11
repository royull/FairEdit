import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.utils import convert
from time import time

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def flipAdj(edge_idx: torch.Tensor,i,j,n):

    # i,j : edit idx
    # n, num of node in graph
    # edge_idx : torch_tensor, shape [m,2]. m = num of existing edges

    # restore the sparse mat
    data = np.ones(edge_idx.shape[1])
    t_mat = coo_matrix((data,edge_idx.cpu().numpy()),shape=(n,n)).tocsr()

    # flip
    if (t_mat[i,j] == 0):
        t_mat[i,j] = 1.
        t_mat[j,i] = 1.
    else:
        t_mat[i,j] = 0.
        t_mat[j,i] = 0.

    # Change back
    return convert.from_scipy_sparse_matrix(t_mat)[0]

class bf_trainer():
    def __init__(self, sense_idx, numEdit, model=None, dataset=None, optimizer=None, features=None, edge_index=None, 
                    labels=None, device=None, train_idx=None, val_idx=None, sens=None):
        self.debug = False
        self.model = model
        self.sense_idx = sense_idx # int, which attribute dimension is sensitive?
        self.numEdit = numEdit # number of edges that we plan to edit
        self.model_name = model.model_name
        self.dataset = dataset
        self.optimizer = optimizer
        self.features = features
        self.edge_index = edge_index
        self.labels = labels
        self.device = device
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.numNode = features.shape[0]
        self.sens = sens
        # From nifty
        counter_features = features.clone()
        counter_features[:, sense_idx] = 1 - counter_features[:, sense_idx]
        self.counter_features = counter_features
    

    def train(self, epochs=200):

        best_loss = 1e5
        minLoss = 1e5
        log_f1 = None
        log_rob = None
        log_fair = None
        log_parity = None
        log_equility = None

        for epoch in range(epochs):
            if self.debug:
                print("===Training Epoch: ", epoch)
            ## TODO: Perform a search over all edge edits within each neighborhood of a node to find one that helps fairness
            # You will need to bring over the fairness metrics (make it be able to use any metric we choose) and then brute force search 

#           One normal step in training
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.edge_index)
            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_train = F.binary_cross_entropy_with_logits(output[self.train_idx], self.labels[self.train_idx].unsqueeze(1).float().to(self.device))
            auc_roc_train = roc_auc_score(self.labels.cpu().numpy()[self.train_idx], output.detach().cpu().numpy()[self.train_idx])
            loss_train.backward()
            self.optimizer.step()

#           Later, just need inference
            self.model.eval()

#           Edit the training graph for the first numEdit steps
#           Save the original graph as basis
            wait = 100

            output = self.model(self.features,self.edge_index.to(self.device))
            preds = (output.squeeze()>0).type_as(self.labels)
                # print(preds)
            counter_output = self.model(self.counter_features.to(self.device),self.edge_index.to(self.device))
            counter_preds = (counter_output.squeeze()>0).type_as(self.labels)
            top_fair_score = 1. - (preds.eq(counter_preds)[self.train_idx].sum().item()/self.train_idx.shape[0])
            if self.debug:
                print("Original Fair Score: ", top_fair_score)
            if (wait <= epoch and epoch < wait + self.numEdit):
                if self.debug:
                    print("Original Fair Score: ", top_fair_score)
                    print("==Try Graph Edit")
                edit_pair = [-1,-1]
                output = self.model(self.features,self.edge_index.to(self.device))
                preds = (output.squeeze()>0).type_as(self.labels)
                # print(preds)
                counter_output = self.model(self.counter_features.to(self.device),self.edge_index.to(self.device))
                counter_preds = (counter_output.squeeze()>0).type_as(self.labels)
                # print(counter_preds)
                # print(preds.eq(counter_preds)[self.train_idx])
                # print(self.train_idx.shape)
                # exit()
                top_fair_score = 1. - (preds.eq(counter_preds)[self.train_idx].sum().item()/self.train_idx.shape[0])
                top_edit = self.edge_index.clone()
                if (self.debug):
                    print("Original Fair Score: ", top_fair_score)
                done_edit = False
                if(top_fair_score == 0 and self.debug):
                    print("Optimal fair score, done!")
                    done_edit = True
                # NOTE: the lower the better
#               Find the best edit
                if (not done_edit):
                    for i in range(self.numNode):
                        if i not in self.train_idx:
                            continue
                        for j in range(i,self.numNode):
                        # for j in range(i,self.numNode):
                            if j not in self.train_idx:
                                continue
                            # Sample every possible one-step edit, calc the counterfactuial fairness, record the best one
                            t1 = time()
                            newGraph = flipAdj(self.edge_index,i,j,self.numNode)
                            t_output = self.model(self.features,newGraph.to(self.device))
                            t_preds = (t_output.squeeze()>0).type_as(self.labels)
                            t_counter_output = self.model(self.counter_features.to(self.device),newGraph.to(self.device))
                            t_counter_preds = (t_counter_output.squeeze()>0).type_as(self.labels)
                            t_fair_score = 1 - (t_preds.eq(t_counter_preds)[self.train_idx].sum().item()/self.train_idx.shape[0])
                            t3 = time()
                            if (self.debug):
                                print("Edit ({},{}), score: {}".format(i,j,t_fair_score))
                                print(t3-t1)
                                print(len(self.train_idx))
                            if (t_fair_score < top_fair_score):
                                top_fair_score = t_fair_score
                                top_edit = newGraph
                                edit_pair = [i,j]
                # Then replace the original with this edit
                #   notice that we only do edit on train_idx, therefore having no effect on val_idx
                self.edge_index = top_edit.to(self.device)
                print("Edit pair: ", edit_pair)

#           Evaluate validation set performance separately,
            output = self.model(self.features, self.edge_index)
             # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_val = F.binary_cross_entropy_with_logits(output[self.val_idx ], self.labels[self.val_idx ].unsqueeze(1).float().to(self.device))

#           F1
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

            if self.debug:
                print("== f1: {} fair: {} robust: {}, parity:{} equility: {}".format(f1_val,fair_score,robustness_score,parity,equality))
            if (loss_train < minLoss):
                minLoss = loss_train
                log_f1 = f1_val
                log_fair = fair_score
                log_rob = robustness_score
                log_parity = parity
                log_equility = equality

#           Record the best model 
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'bruteforce', self.dataset))
        print("===\n Final: f1: {} fair: {} robust: {}, parity:{} equility: {}".format(log_f1,log_fair,log_rob,log_parity,log_equility))

