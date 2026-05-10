"""Model Service for training and inference"""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

logger = logging.getLogger(__name__)

class SimpleConvNet(nn.Module):
    """Simple CNN for medical image classification"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ModelService:
    """Handles model training and evaluation"""
    
    def __init__(self, device: torch.device, num_classes: int = 2, in_channels: int = 3):
        self.device = device
        self.model = SimpleConvNet(in_channels=in_channels, num_classes=num_classes)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.metrics = {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "test_auc": 0.0,
            "total_queries": 0,
        }
        
        self.confusion_matrix = None
        self.roc_curve_data = None
        
    def train_step(
        self,
        batch_images: torch.Tensor,
        batch_labels: torch.Tensor,
        epochs: int = 1
    ):
        """Train model for one or more epochs"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Mini-batches
            batch_size = 32
            for i in range(0, len(batch_images), batch_size):
                batch_x = batch_images[i:i+batch_size].to(self.device)
                batch_y = batch_labels[i:i+batch_size].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        accuracy = correct / total if total > 0 else 0
        self.metrics["train_accuracy"] = accuracy
        
        logger.info(f"Training epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={accuracy:.4f}")
    
    def evaluate(
        self,
        test_images: torch.Tensor,
        test_labels: torch.Tensor
    ):
        """Evaluate model on test set"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(test_images.to(self.device))
            predictions = torch.argmax(outputs, dim=1)
            confidences = F.softmax(outputs, dim=1)
        
        predictions = predictions.cpu().numpy()
        confidences = confidences.cpu().numpy()
        test_labels = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels
        
        # Accuracy
        accuracy = (predictions == test_labels).mean()
        self.metrics["test_accuracy"] = float(accuracy)
        
        # Confusion Matrix
        self.confusion_matrix = confusion_matrix(test_labels, predictions)
        
        # ROC Curve (for binary classification)
        if len(np.unique(test_labels)) == 2:
            fpr, tpr, _ = roc_curve(test_labels, confidences[:, 1])
            roc_auc = auc(fpr, tpr)
            self.metrics["test_auc"] = float(roc_auc)
            self.roc_curve_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc),
            }
        
        logger.info(f"Evaluation: Accuracy={accuracy:.4f}, AUC={self.metrics['test_auc']:.4f}")
    
    def predict(self, images: torch.Tensor) -> Dict:
        """Get predictions for images"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            confidences = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        return {
            "predictions": predictions.cpu().numpy().tolist(),
            "confidences": confidences.cpu().numpy().tolist(),
            "uncertainties": (1 - confidences.max(dim=1)[0]).cpu().numpy().tolist(),
        }
    
    def get_uncertainty(self, image: torch.Tensor) -> float:
        """Get prediction uncertainty (entropy)"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(image.unsqueeze(0).to(self.device))
            probs = F.softmax(output, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        
        return entropy.item()
    
    def get_metrics(self) -> Dict:
        """Get current model metrics"""
        return self.metrics.copy()
    
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix"""
        return self.confusion_matrix
    
    def get_roc_curve(self) -> Optional[Dict]:
        """Get ROC curve data"""
        return self.roc_curve_data or {"fpr": [], "tpr": [], "auc": 0.0}
    
    def get_weights(self) -> Dict:
        """Get model weights for transfer"""
        return {name: param.cpu().detach().numpy() for name, param in self.model.named_parameters()}
    
    def load_weights(self, weights: Dict):
        """Load model weights"""
        state_dict = {}
        for name, array in weights.items():
            state_dict[name] = torch.from_numpy(array)
        self.model.load_state_dict(state_dict)
