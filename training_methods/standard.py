
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score

from utils import fair_metric

class standard_trainer():
    def __init__(self, sense_idx,model=None,  dataset=None, optimizer=None, features=None, edge_index=None, 
                    labels=None, device=None, train_idx=None, val_idx=None,sens=None):
        self.model = model
        self.model_name = model.model_name
        self.dataset = dataset
        self.optimizer = optimizer
        self.features = features
        self.edge_index = edge_index
        self.labels = labels
        self.device = device
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.sens = sens
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
            print("===Training Epoch: ", epoch)
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

            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'standard', self.dataset))
            
            f1_val = f1_score(self.labels[self.val_idx ].cpu().numpy(), preds[self.val_idx ].cpu().numpy())
            print(f1_val)
        auc_roc_val = roc_auc_score(self.labels.cpu().numpy()[self.val_idx ], output.detach().cpu().numpy()[self.val_idx ])
        f1_val = f1_score(self.labels[self.val_idx ].cpu().numpy(), preds[self.val_idx ].cpu().numpy())
        # parity, equality = fair_metric(preds,self.labels,self.sens)
        counter_output = self.model(self.counter_features.to(self.device),self.edge_index.to(self.device))
        counter_preds = (counter_output.squeeze()>0).type_as(self.labels)
        fair_score = 1 - (preds.eq(counter_preds)[self.val_idx].sum().item()/self.val_idx.shape[0])
        print("== f1: {} fair: {}".format(f1_val,fair_score))

        # return auc_roc_val,f1_val,parity,equality
        return auc_roc_val,f1_val,1,1
