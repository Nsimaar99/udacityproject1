import argparse
import torch
import torch.nn as nn
from PIL import Image
import json
import numpy as np
from torchvision import models, transforms

# Define the ImagePredictor class
class ImagePredictor:
    def __init__(self, model, json_file_path='cat_to_name.json'):
        self.model = model
        self.label_mapping = self.load_label_mapping(json_file_path)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_label_mapping(self, json_file_path):
        # Load the label mapping from JSON
        with open(json_file_path, 'r') as f:
            return json.load(f)

    def get_flower_name(self, class_index):
        # Retrieve flower name based on class index
        return self.label_mapping.get(str(class_index), "Unknown class")

    def process_image(self, image_path):
        # Open the image and ensure it is in RGB mode
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)
        return image

    def predict(self, image_path, topk=5):
        # Ensure that the model is on the correct device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Preprocess the image and convert it to a tensor
        image_tensor = self.process_image(image_path).unsqueeze(0)
        
        # Move the input tensor to the same device as the model
        image_tensor = image_tensor.to(device)
        
        # Disable gradient calculation for inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Apply softmax to get probabilities
        output = torch.softmax(output, dim=1)
        
        # Get the top K probabilities and class indices
        top_probs, top_indices = torch.topk(output, topk)
        
        # Convert probabilities and indices to CPU if needed
        top_probs = top_probs.cpu().numpy().squeeze()
        top_indices = top_indices.cpu().numpy().squeeze()
        
        return top_probs, top_indices

    def display_predictions(self, image_path, topk=5):
        top_probs, top_indices = self.predict(image_path, topk)
        print(f"Top {topk} Predictions:")
        for i in range(len(top_probs)):
            class_name = self.get_flower_name(top_indices[i])
            print(f"Flower name: {class_name}, Probability: {top_probs[i]:.4f}")

def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')  # Load on CPU
    
    # Load the ResNet50 model with no pre-trained weights
    model = models.resnet50(weights=None)  # Alternatively use pre-trained weights if needed
    num_classes = len(checkpoint['class_to_idx'])
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=num_classes)
    )

    # Load the model state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  

    return model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    # Determine device to use (GPU or CPU)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    # Load the model from the checkpoint
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    # Create ImagePredictor instance and display predictions
    image_predictor = ImagePredictor(model, args.category_names)
    image_predictor.display_predictions(args.image_path, topk=args.top_k)

if __name__ == '__main__':
    main()

