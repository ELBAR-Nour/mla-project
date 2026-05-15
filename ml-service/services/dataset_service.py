import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO
import base64
from io import BytesIO
from PIL import Image

class DatasetService:
    def __init__(self, device):
        self.device = device
        self.dataset_name = None
        self.split = None
        self.n_classes = None
        self.task = None
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.val_loader = None
        self.test_loader = None
        self.info = None

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def load_dataset(self, dataset_name="pneumoniamnist", split="train", sample_size=None):
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.info = INFO[self.dataset_name]
        self.n_classes = len(self.info['label'])
        self.task = self.info['task']
        DataClass = getattr(medmnist, self.info['python_class'])
        transform = self.get_transforms()
        root = './data'
        os.makedirs(root, exist_ok=True)
        
        train_ds = DataClass(split='train', transform=transform, download=True, root=root)
        val_ds   = DataClass(split='val',   transform=transform, download=True, root=root)
        test_ds  = DataClass(split='test',  transform=transform, download=True, root=root)
        
        self.train_images, self.train_labels = self._extract_arrays(train_ds)
        self.val_images, self.val_labels = self._extract_arrays(val_ds)
        self.test_images, self.test_labels = self._extract_arrays(test_ds)

        if sample_size is not None and sample_size < len(self.train_images):
            indices = np.random.choice(len(self.train_images), sample_size, replace=False)
            self.train_images = self.train_images[indices]
            self.train_labels = self.train_labels[indices]
            
        self.val_loader = self._make_loader(self.val_images, self.val_labels, shuffle=False)
        self.test_loader = self._make_loader(self.test_images, self.test_labels, shuffle=False)
        
        self._initialise_pools(len(self.train_images), 100)

    def _extract_arrays(self, dataset):
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        images_list, labels_list = [], []
        for imgs, labels in loader:
            images_list.append(imgs.numpy())
            labels_list.append(labels.numpy().squeeze())
        return np.concatenate(images_list), np.concatenate(labels_list)

    def _initialise_pools(self, n_total: int, n_labeled: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        all_idx = list(range(n_total))
        rng.shuffle(all_idx)
        self.labeled_idx = all_idx[:n_labeled]
        self.unlabeled_idx = all_idx[n_labeled:]
        return self.labeled_idx, self.unlabeled_idx

    def _make_loader(self, images, labels, indices=None, batch_size=64, shuffle=True):
        if indices is not None:
            imgs = torch.tensor(images[indices], dtype=torch.float32)
            lbs  = torch.tensor(labels[indices], dtype=torch.long)
        else:
            imgs = torch.tensor(images, dtype=torch.float32)
            lbs  = torch.tensor(labels, dtype=torch.long)
        ds = TensorDataset(imgs, lbs)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=self.device.type == 'cuda')
        
    def get_dataset_info(self):
        if self.train_images is None:
            return {"total_samples": 0, "image_shape": []}
        return {
            "name": self.dataset_name,
            "split": self.split,
            "total_samples": len(self.train_images),
            "test_samples": len(self.test_images),
            "image_shape": list(self.train_images.shape[1:]),
            "n_classes": self.n_classes,
            "label_name": "Label",
            "labeled_count": len(self.labeled_idx),
            "unlabeled_count": len(self.unlabeled_idx)
        }

    def _image_to_base64(self, img_array):
        img_array = img_array.squeeze()
        img_array = (img_array * 0.5 + 0.5) * 255
        img_array = img_array.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def get_sample(self, image_id):
        img = self.train_images[image_id]
        base64_str = self._image_to_base64(img)
        return {
            "image_base64": base64_str,
            "shape": list(img.shape)
        }

    def get_batch(self, count):
        indices = random.sample(self.unlabeled_idx, min(count, len(self.unlabeled_idx)))
        samples = []
        for idx in indices:
            samples.append({
                "image_id": idx,
                "image_base64": self._image_to_base64(self.train_images[idx]),
                "shape": list(self.train_images[idx].shape)
            })
        return samples

    def get_labeled_loader(self, batch_size=64):
        return self._make_loader(self.train_images, self.train_labels, self.labeled_idx, batch_size=batch_size, shuffle=True)
