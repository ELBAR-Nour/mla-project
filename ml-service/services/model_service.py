import base64
import copy
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve

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
        self.active_model_name = None
        self.latest_metrics = {}
        self.latest_probs = None
        self.latest_labels = None
        service_root = Path(__file__).resolve().parents[1]
        self.models_dir = Path(os.getenv("ML_MODELS_DIR", service_root / "models"))

    def initialize_model(self, n_classes):
        self.n_classes = n_classes
        self.model = MedicalCNN(n_classes, CONFIG['dropout_rate']).to(self.device)
        self.active_model_name = None

    def _load_torch_file(self, path: Path):
        return torch.load(path, map_location="cpu")

    def _classifier_state_dict(self, artifact) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(artifact, dict):
            return None
        if "state_dict" in artifact and isinstance(artifact["state_dict"], dict):
            artifact = artifact["state_dict"]
        if not artifact:
            return None
        if not all(hasattr(value, "shape") for value in artifact.values()):
            return None
        if any(key.startswith("features.") or key.startswith("classifier.") for key in artifact.keys()):
            return artifact
        return None

    def _infer_classifier_metadata(self, state_dict: Dict[str, torch.Tensor]) -> Dict:
        first_conv = state_dict.get("features.0.weight")
        classifier_weights = [
            (key, value)
            for key, value in state_dict.items()
            if key.startswith("classifier.") and key.endswith(".weight") and len(value.shape) == 2
        ]
        if not classifier_weights:
            raise ValueError("Could not infer classifier output layer from artifact.")

        classifier_weights.sort(key=lambda item: int(item[0].split(".")[1]))
        output_weight = classifier_weights[-1][1]
        return {
            "n_classes": int(output_weight.shape[0]),
            "in_channels": int(first_conv.shape[1]) if first_conv is not None else 1,
            "architecture": "MedicalCNN",
        }

    def _resolve_model_path(self, model_name: str) -> Path:
        requested = Path(model_name).name
        candidates = [requested]
        if not requested.endswith(".pt"):
            candidates.append(f"{requested}.pt")

        for candidate in candidates:
            path = self.models_dir / candidate
            if path.exists() and path.is_file():
                return path
        raise FileNotFoundError(f"Model artifact '{model_name}' was not found in {self.models_dir}.")

    def list_model_artifacts(self) -> List[Dict]:
        artifacts = []
        if not self.models_dir.exists():
            return artifacts

        for path in sorted(self.models_dir.glob("*.pt")):
            try:
                loaded = self._load_torch_file(path)
                state_dict = self._classifier_state_dict(loaded)
                if state_dict is None:
                    continue
                metadata = self._infer_classifier_metadata(state_dict)
                strategy = path.stem.split("_model_")[-1] if "_model_" in path.stem else "classifier"
                artifacts.append({
                    "name": path.name,
                    "strategy": strategy,
                    "n_classes": metadata["n_classes"],
                    "in_channels": metadata["in_channels"],
                    "architecture": metadata["architecture"],
                    "size_bytes": path.stat().st_size,
                    "active": path.name == self.active_model_name,
                })
            except Exception as exc:
                artifacts.append({
                    "name": path.name,
                    "error": str(exc),
                    "size_bytes": path.stat().st_size,
                    "active": False,
                })
        return artifacts

    def _default_model_name(self) -> str:
        artifacts = [artifact for artifact in self.list_model_artifacts() if "error" not in artifact]
        if not artifacts:
            raise FileNotFoundError(f"No classifier .pt artifacts found in {self.models_dir}.")
        preferred = next((artifact for artifact in artifacts if artifact["strategy"] == "entropy"), None)
        return (preferred or artifacts[0])["name"]

    def load_model_artifact(self, model_name: str) -> Dict:
        path = self._resolve_model_path(model_name)
        loaded = self._load_torch_file(path)
        state_dict = self._classifier_state_dict(loaded)
        if state_dict is None:
            raise ValueError(f"Artifact '{path.name}' is not a MedicalCNN classifier state dict.")

        metadata = self._infer_classifier_metadata(state_dict)
        model = MedicalCNN(
            metadata["n_classes"],
            CONFIG["dropout_rate"],
            in_channels=metadata["in_channels"],
        ).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.model = model
        self.n_classes = metadata["n_classes"]
        self.active_model_name = path.name
        return {
            "status": "loaded",
            "name": path.name,
            **metadata,
        }

    def ensure_model_loaded(self, model_name: Optional[str] = None):
        if model_name and model_name != self.active_model_name:
            self.load_model_artifact(model_name)
        elif self.model is None:
            self.load_model_artifact(self._default_model_name())
        return self.model

    def tensor_from_base64(self, image_base64: str) -> torch.Tensor:
        if "," in image_base64 and image_base64.lstrip().startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        raw = base64.b64decode(image_base64)
        image = Image.open(BytesIO(raw)).convert("L").resize((28, 28))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def predict_tensor(
        self,
        image_tensor,
        model_name: Optional[str] = None,
        true_label: Optional[int] = None,
        class_labels: Optional[List[str]] = None,
    ) -> Dict:
        model = self.ensure_model_loaded(model_name)

        if not torch.is_tensor(image_tensor):
            image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        predicted_label = int(np.argmax(probs))
        confidence = float(np.max(probs))
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        if class_labels is None:
            class_labels = [str(i) for i in range(len(probs))]

        return {
            "model_name": self.active_model_name,
            "predicted_label": predicted_label,
            "predicted_label_name": class_labels[predicted_label] if predicted_label < len(class_labels) else str(predicted_label),
            "true_label": true_label,
            "true_label_name": class_labels[true_label] if true_label is not None and true_label < len(class_labels) else None,
            "probabilities": [float(prob) for prob in probs],
            "class_labels": class_labels,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "correct": bool(predicted_label == true_label) if true_label is not None else None,
        }

    def compute_ece(self, loader, n_bins: int = 10) -> float:
        if self.model is None or loader is None:
            return 0.0

        self.model.eval()
        confidences, accuracies = [], []
        with torch.no_grad():
            for images, labels in loader:
                probs = F.softmax(self.model(images.to(self.device)), dim=-1).cpu().numpy()
                conf = probs.max(axis=1)
                pred = probs.argmax(axis=1)
                acc = (pred == labels.numpy().flatten()).astype(float)
                confidences.extend(conf)
                accuracies.extend(acc)

        conf_arr = np.array(confidences)
        acc_arr = np.array(accuracies)
        if len(conf_arr) == 0:
            return 0.0

        ece = 0.0
        bins = np.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf_arr >= lo) & (conf_arr < hi)
            if mask.sum() > 0:
                ece += mask.sum() * abs(conf_arr[mask].mean() - acc_arr[mask].mean())
        return float(ece / len(conf_arr))

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
        metrics = {
            'acc': acc, 'auc': auc,
            'f1': f1_score(labels_arr, preds_arr, average=avg, zero_division=0),
            'recall': recall_score(labels_arr, preds_arr, average=avg, zero_division=0),
            'precision': precision_score(labels_arr, preds_arr, average=avg, zero_division=0),
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
        labels = list(range(self.n_classes or 2))
        return confusion_matrix(self.latest_labels, preds, labels=labels)

    def get_roc_curve(self):
        if self.latest_probs is None or self.latest_labels is None or self.n_classes != 2:
            return {"fpr": [], "tpr": [], "thresholds": []}
        fpr, tpr, thresholds = roc_curve(self.latest_labels, self.latest_probs[:, 1])
        return {
            "fpr": [float(value) for value in fpr],
            "tpr": [float(value) for value in tpr],
            "thresholds": [
                float(value) if np.isfinite(value) else None
                for value in thresholds
            ],
        }
