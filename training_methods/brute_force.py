import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

class bf_trainer():
    def __init__(self, sense_idx,model=None, dataset=None, optimizer=None, features=None, edge_index=None, 
                    labels=None, device=None, train_idx=None, val_idx=None, fair_metric = None):
        self.model = model
        self.sense_idx = sense_idx # int, which attribute dimension is sensitive?
        self.model_name = model.model_name
        self.dataset = dataset
        self.optimizer = optimizer
        self.features = features
        self.edge_index = edge_index
        self.labels = labels
        self.device = device
        self.train_idx = train_idx
        self.val_idx = val_idx
    

    def Compute_Fair(self):
        # Returns: Counterfactual fainess score, Robusness fairness score
        # Prepare
        idx_test = self.val_idx
        # Mainly from nifty_sota_gnn.py, line 333-341
        self.model.eval()
        output = self.model(self.features.to(self.device), self.edge_index.to(self.device))
        counter_features = self.features.clone
        counter_features[:, self.sens_idx] = 1 - counter_features[:, self.sens_idx]
        counter_output = self.model(counter_features.to(self.devices), self.edge_index.to(self.device))
        noisy_features = self.features.clone() + torch.ones(self.features.shape).normal_(0, 1).to(self.device)
        noisy_output = self.model(noisy_features.to(self.device), self.edge_index.to(self.device))
        # Calc
        # Mainly from nifty_sota_gnn.py, line 366-374
        output_preds = (output.squeeze()>0).type_as(self.labels)
        counter_output_preds = (counter_output.squeeze()>0).type_as(self.labels)
        noisy_output_preds = (noisy_output.squeeze()>0).type_as(self.labels)
        auc_roc_test = roc_auc_score(self.labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
        counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
        robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), self.labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
        f1_s = f1_score(self.labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
        return parity, counterfactual_fairness, robustness_score, auc_roc_test, 


    def train(self, epochs=200):

        best_loss = 100
        best_acc = 0

        for epoch in range(epochs):
            ## TODO: Perform a search over all edge edits within each neighborhood of a node to find one that helps fairness
            # You will need to bring over the fairness metrics (make it be able to use any metric we choose) and then brute force search 
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.edge_index)

            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_train = F.binary_cross_entropy_with_logits(output[self.train_idx], self.labels[self.train_idx].unsqueeze(1).float().to(self.device))

            auc_roc_train = roc_auc_score(self.labels.cpu().numpy()[self.train_idx], output.detach().cpu().numpy()[self.train_idx])
            loss_train.backward()
            self.optimizer.step()

            # Evaluate validation set performance separately,
            self.model.eval()
            output = self.model(self.features, self.edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_val = F.binary_cross_entropy_with_logits(output[self.val_idx ], self.labels[self.val_idx ].unsqueeze(1).float().to(self.device))

            auc_roc_val = roc_auc_score(self.labels.cpu().numpy()[self.val_idx ], output.detach().cpu().numpy()[self.val_idx ])
            f1_val = f1_score(self.labels[self.val_idx ].cpu().numpy(), preds[self.val_idx ].cpu().numpy())

            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'bruteforce', self.dataset))

