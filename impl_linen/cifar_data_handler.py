"""
CIFAR-100 Data Handler for Incremental Learning Experiments
Handles all data loading, preprocessing, and partitioning operations.
"""

import os
from typing import List, Callable, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms

from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import (
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotator,
    ToTensor,
)


def subsample_cifar_data_set(sub_sample_indices: torch.Tensor, cifar_data: CifarDataSet):
    """In-place sub-sampling identical to the PyTorch version."""
    idx = sub_sample_indices.cpu().numpy()
    cifar_data.data["data"] = cifar_data.data["data"][idx]
    cifar_data.data["labels"] = cifar_data.data["labels"][idx]
    cifar_data.integer_labels = torch.as_tensor(cifar_data.integer_labels)[idx].tolist()
    cifar_data.current_data = cifar_data.partition_data()


class CIFARDataHandler:
    """Handles all CIFAR-100 data operations for incremental learning."""
    
    def __init__(self, data_path: str, num_workers: int = 1, random_seed: int = 42):
        self.data_path = data_path
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # Constants
        self.num_classes = 100
        self.num_images_per_class = 450
        self.batch_sizes = {"train": 90, "validation": 50, "test": 100}
        
        # CIFAR-100 normalization values
        self.cifar_mean = (0.5071, 0.4865, 0.4409)
        self.cifar_std = (0.2673, 0.2564, 0.2762)
        
        # Generate random class order for continual learning
        np.random.seed(self.random_seed)
        self.all_classes = np.random.permutation(self.num_classes)
        
    def get_transformations(self, validation: bool = False) -> List[Callable]:
        """Get data transformations for training or validation."""
        transformations = [
            ToTensor(swap_color_axis=True),
            Normalize(mean=self.cifar_mean, std=self.cifar_std),
        ]
        
        if not validation:
            # Add augmentations for training
            transformations += [
                RandomHorizontalFlip(p=0.5),
                RandomCrop(size=32, padding=4, padding_mode="reflect"),
                RandomRotator(degrees=(0, 15)),
            ]
            
        return transformations
    
    def get_validation_and_train_indices(self, cifar_data: CifarDataSet) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split data into train/validation matching PyTorch/NNX versions."""
        num_val_samples_per_class = 50
        num_train_samples_per_class = 450
        validation_set_size = 5000
        train_set_size = 45000

        validation_indices = torch.zeros(validation_set_size, dtype=torch.int32)
        train_indices = torch.zeros(train_set_size, dtype=torch.int32)
        current_val_samples = 0
        current_train_samples = 0
        
        for i in range(self.num_classes):
            class_indices = torch.where(torch.tensor(cifar_data.data["labels"][:, i]) == 1)[0]
            validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] = class_indices[:num_val_samples_per_class]
            train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] = class_indices[num_val_samples_per_class:]
            current_val_samples += num_val_samples_per_class
            current_train_samples += num_train_samples_per_class

        return train_indices, validation_indices
    
    def create_cifar_dataset(self, train: bool = True, validation: bool = False) -> CifarDataSet:
        """Create a CIFAR dataset with appropriate transformations."""
        cifar_data = CifarDataSet(
            root_dir=self.data_path,
            train=train,
            cifar_type=100,
            device=None,
            image_normalization="max",
            label_preprocessing="one-hot",
            use_torch=True,
        )
        
        transformations = self.get_transformations(validation=validation)
        cifar_data.set_transformation(torchvision.transforms.Compose(transformations))
        
        return cifar_data
    
    def get_data(self, train: bool = True, validation: bool = False) -> Tuple[CifarDataSet, DataLoader]:
        """Loads CIFAR-100 via CifarDataSet and returns (dataset, dataloader)."""
        cifar_data = self.create_cifar_dataset(train=train, validation=validation)
        
        if not train:
            # Test set
            batch_size = self.batch_sizes["test"]
            dataloader = DataLoader(
                cifar_data, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=self.num_workers
            )
            return cifar_data, dataloader

        # Train/validation split
        train_indices, val_indices = self.get_validation_and_train_indices(cifar_data)
        indices = val_indices if validation else train_indices
        subsample_cifar_data_set(indices, cifar_data)
        
        batch_size = self.batch_sizes["validation"] if validation else self.batch_sizes["train"]
        dataloader = DataLoader(
            cifar_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
        
        return cifar_data, dataloader
    
    def select_classes_for_datasets(self, current_num_classes: int, *datasets: CifarDataSet):
        """Select current classes for all provided datasets."""
        for dataset in datasets:
            dataset.select_new_partition(self.all_classes[:current_num_classes])
    
    def get_class_order(self) -> np.ndarray:
        """Get the random class order for this experiment."""
        return self.all_classes.copy()