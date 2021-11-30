
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score

class fair_edit_trainer():
    def __init__(self, model=None, dataset=None, optimizer=None, features=None, edge_index=None, 
                    labels=None, device=None, train_idx=None, val_idx=None):
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

