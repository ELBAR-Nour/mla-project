import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, matthews_corrcoef

CONFIG = {
    "cnn_epochs": 15,
    "cnn_lr": 1e-3,
    "cnn_weight_decay": 1e-4,
    "dropout_rate": 0.3,
}

class MedicalCNN(nn.Module):
    def __init__(self, n_classes: int, dropout_rate: float = 0.3, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    def predict_proba(self, x):
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

class ModelService:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.n_classes = None
        self.latest_metrics = {}
        self.latest_probs = None
        self.latest_labels = None

    def initialize_model(self, n_classes):
        self.n_classes = n_classes
        self.model = MedicalCNN(n_classes, CONFIG['dropout_rate']).to(self.device)

    def train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(loader))

    def evaluate(self, model, loader, n_classes, return_raw=False):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                probs = F.softmax(model(images.to(self.device)), dim=-1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())
        if not all_probs:
            return {'acc': 0, 'auc': 0, 'f1': 0, 'recall': 0, 'precision': 0}

        probs_arr = np.concatenate(all_probs)
        labels_arr = np.concatenate(all_labels)
        preds_arr = probs_arr.argmax(axis=1)
        acc = accuracy_score(labels_arr, preds_arr)
        
        if n_classes == 2:
            try:
                auc = roc_auc_score(labels_arr, probs_arr[:, 1])
            except ValueError:
                auc = 0.5
        else:
            try:
                auc = roc_auc_score(labels_arr, probs_arr, multi_class='ovr', average='macro')
            except ValueError:
                auc = 0.5
                
        avg = 'binary' if n_classes == 2 else 'macro'
        
        specificity = 0.0
        if n_classes == 2:
            tn, fp, fn, tp = confusion_matrix(labels_arr, preds_arr, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
        metrics = {
            'acc': acc, 'auc': auc,
            'f1': f1_score(labels_arr, preds_arr, average=avg, zero_division=0),
            'recall': recall_score(labels_arr, preds_arr, average=avg, zero_division=0),
            'precision': precision_score(labels_arr, preds_arr, average=avg, zero_division=0),
            'specificity': float(specificity),
            'mcc': float(matthews_corrcoef(labels_arr, preds_arr))
        }
        
        if return_raw:
            return metrics, probs_arr, labels_arr
        return metrics

    def train_model(self, model, train_loader, val_loader, n_classes, epochs=None, lr=None):
        epochs = epochs or CONFIG['cnn_epochs']
        lr = lr or CONFIG['cnn_lr']
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG['cnn_weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        best_auc, best_state, patience_ctr = 0.0, None, 0
        
        for epoch in range(epochs):
            self.train_epoch(model, train_loader, optimizer, criterion)
            metrics = self.evaluate(model, val_loader, n_classes)
            auc = metrics['auc']
            scheduler.step()
            if auc > best_auc:
                best_auc = auc
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= 5:
                break
                
        if best_state:
            model.load_state_dict(best_state)
            
        metrics, probs, labels = self.evaluate(model, val_loader, n_classes, return_raw=True)
        self.latest_metrics = metrics
        self.latest_probs = probs
        self.latest_labels = labels
        self.model = model
        return best_auc

    def get_metrics(self):
        return self.latest_metrics

    def get_confusion_matrix(self):
        if self.latest_probs is None or self.latest_labels is None:
            return np.zeros((self.n_classes or 2, self.n_classes or 2))
        preds = self.latest_probs.argmax(axis=1)
        return confusion_matrix(self.latest_labels, preds)

    def get_roc_curve(self):
        if self.latest_probs is None or self.latest_labels is None or self.n_classes != 2:
            return {"fpr": [], "tpr": [], "thresholds": []}
        fpr, tpr, thresholds = roc_curve(self.latest_labels, self.latest_probs[:, 1])
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }
