
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from utils.gnn_explainer import GNNExplainer

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

    def fair_loss(self):
        
        grad_gen = GNNExplainer(self.model)

        edge_mask = grad_gen.explain_graph(self.features, self.edge_index)
        print(edge_mask)
        sys.exit()

    def train(self, epochs=200):

        best_loss = 100
        best_acc = 0

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.edge_index)

            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_train = F.binary_cross_entropy_with_logits(output[self.train_idx], self.labels[self.train_idx].unsqueeze(1).float().to(self.device))

            loss_train.backward()
            self.optimizer.step()

       
            
            # Evaluate validation set performance separately,
            self.model.eval()
            output = self.model(self.features, self.edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(self.labels)
            loss_val = F.binary_cross_entropy_with_logits(output[self.val_idx ], self.labels[self.val_idx ].unsqueeze(1).float().to(self.device))

            self.fair_loss()
            sys.exit()
        
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), 'results/weights/{0}_{1}_{2}.pt'.format(self.model_name, 'fairedit', self.dataset))

