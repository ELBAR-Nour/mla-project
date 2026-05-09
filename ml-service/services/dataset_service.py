"""Dataset Service for managing medical imaging data"""
import base64
import io
import logging
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import medmnist
from medmnist import INFO

logger = logging.getLogger(__name__)

class DatasetService:
    """Manages dataset loading and sampling"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.dataset = None
        self.test_dataset = None
        self.dataset_name = None
        self.dataset_info = None
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        
    def load_dataset(
        self,
        dataset_name: str = "PneumoniaMNIST",
        split: str = "train",
        sample_size: Optional[int] = None
    ):
        """Load medical dataset from MedMNIST"""
        try:
            self.dataset_name = dataset_name
            data_flag = dataset_name
            download = True
            
            # Load dataset
            DataClass = getattr(medmnist, dataset_name)
            self.dataset = DataClass(split=split, transform=None, download=download)
            self.test_dataset = DataClass(split="test", transform=None, download=download)
            
            # Get info
            info = INFO[data_flag]
            self.dataset_info = {
                "name": dataset_name,
                "split": split,
                "total_samples": len(self.dataset),
                "test_samples": len(self.test_dataset),
                "image_shape": (28, 28, 3) if info["n_channels"] == 3 else (28, 28),
                "n_classes": info["n_classes"],
                "label_name": info["label"],
            }
            
            if sample_size and sample_size < len(self.dataset):
                indices = np.random.choice(len(self.dataset), sample_size, replace=False)
                self.unlabeled_indices = set(indices)
            else:
                self.unlabeled_indices = set(range(len(self.dataset)))
            
            logger.info(f"✅ Loaded {dataset_name}: {len(self.dataset)} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def get_dataset_info(self) -> Dict:
        """Get dataset metadata"""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        
        return {
            **self.dataset_info,
            "labeled_count": len(self.labeled_indices),
            "unlabeled_count": len(self.unlabeled_indices),
        }
    
    def get_sample(self, image_id: int) -> Dict:
        """Get a single sample as base64 image"""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        
        if image_id >= len(self.dataset):
            raise ValueError(f"Image ID {image_id} out of bounds")
        
        image, label = self.dataset[image_id]
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image_id": image_id,
            "image_base64": image_base64,
            "shape": list(image.size) if hasattr(image, 'size') else [28, 28],
            "label": int(label) if isinstance(label, (int, np.integer)) else int(label[0]),
            "is_labeled": image_id in self.labeled_indices,
        }
    
    def get_batch(self, count: int = 10) -> List[Dict]:
        """Get a batch of unlabeled samples"""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        
        # Get random unlabeled indices
        available = list(self.unlabeled_indices)
        sample_indices = np.random.choice(
            available,
            size=min(count, len(available)),
            replace=False
        )
        
        batch = []
        for idx in sample_indices:
            batch.append(self.get_sample(idx))
        
        return batch
    
    def mark_labeled(self, image_id: int):
        """Mark a sample as labeled"""
        if image_id in self.unlabeled_indices:
            self.unlabeled_indices.remove(image_id)
            self.labeled_indices.add(image_id)
    
    def get_tensor(self, image_id: int) -> torch.Tensor:
        """Get sample as torch tensor"""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        
        image, _ = self.dataset[image_id]
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        else:
            image = image.float()
        
        # Normalize to [0, 1] if needed
        if image.max() > 1:
            image = image / 255.0
        
        # Add batch dimension if needed
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        return image.to(self.device)
