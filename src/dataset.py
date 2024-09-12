import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os

class FlowerDataset(Dataset):
    def __init__(self, data_dir, batch_size=64):
        """
        Custom dataset class for loading and transforming the flower dataset.

        Parameters:
        - data_dir (str): The root directory where the dataset is stored.
        - batch_size (int): The batch size for data loading.
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.valid_dir = os.path.join(self.data_dir, 'valid')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.batch_size = batch_size

        # Define transforms for train, validation, and test datasets
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # Load datasets using ImageFolder
        self.image_datasets = {
            'train': datasets.ImageFolder(self.train_dir, transform=self.data_transforms['train']),
            'valid': datasets.ImageFolder(self.valid_dir, transform=self.data_transforms['valid']),
            'test': datasets.ImageFolder(self.test_dir, transform=self.data_transforms['test']),
        }

        # Create DataLoaders for each dataset
        self.dataloaders = {
            'train': DataLoader(self.image_datasets['train'], batch_size=self.batch_size, shuffle=True),
            'valid': DataLoader(self.image_datasets['valid'], batch_size=self.batch_size, shuffle=False),
            'test': DataLoader(self.image_datasets['test'], batch_size=self.batch_size, shuffle=False)
        }

    def get_dataloaders(self):
        """
        Returns the train, validation, and test dataloaders.
        """
        return self.dataloaders['train'], self.dataloaders['valid'], self.dataloaders['test']

    def get_num_classes(self):
        """
        Returns the number of classes in the dataset.
        """
        return len(self.image_datasets['train'].class_to_idx)
