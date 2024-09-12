import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class FlowerResNet50:
    def __init__(self, num_classes=102, learning_rate=1e-3):
        """
        Initializes the ResNet50 model, modifies the classifier, and sets up the optimizer and loss function.
        
        Parameters:
        - num_classes (int): The number of classes for the dataset. Default is 102.
        - learning_rate (float): The learning rate for the optimizer. Default is 1e-3.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._initialize_model(num_classes)
        self.model.to(self.device)
        
        # Define the loss function (criterion)
        self.criterion = nn.CrossEntropyLoss()

        # Define the optimizer (only for the final classifier layers)
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)

    def _initialize_model(self, num_classes):
        """
        Loads a pre-trained ResNet50 model and modifies the final fully connected layer.
        
        Parameters:
        - num_classes (int): The number of classes for the final classification layer.
        
        Returns:
        - model: Modified ResNet50 model.
        """
        # Load the pre-trained ResNet50 model
        model = models.resnet50(weights='DEFAULT')

        # Modify the fully connected layer to match the number of classes
        model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        )

        return model

    def print_model_architecture(self):
        """Prints the architecture of the modified ResNet50 model."""
        print(self.model)

    def get_model(self):
        """Returns the model."""
        return self.model

    def get_criterion(self):
        """Returns the loss function (criterion)."""
        return self.criterion

    def get_optimizer(self):
        """Returns the optimizer."""
        return self.optimizer

