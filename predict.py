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
        with open(json_file_path, 'r') as f:
            return json.load(f)

    def get_flower_name(self, class_index):
        return self.label_mapping.get(str(class_index), "Unknown class")

    def process_image(self, image):
        # Ensure the image is in RGB mode
        image = image.convert('RGB')
        image = self.preprocess(image)
        return image

    def predict(self, image_path, topk=5):
        image = Image.open(image_path)
        image_tensor = self.process_image(image).unsqueeze(0)  # Add batch dimension
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, topk)
            top_probs = top_probs.squeeze().cpu().numpy()
            top_indices = top_indices.squeeze().cpu().numpy()
        return top_probs, top_indices

    def display_predictions(self, image_path, topk=5):
        top_probs, top_indices = self.predict(image_path, topk)
        print(f"Top {topk} Predictions:")
        for i in range(len(top_probs)):
            class_name = self.get_flower_name(top_indices[i])
            print(f"Flower name: {class_name}, Probability: {top_probs[i]:.4f}")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')  # Load on CPU
    
    model = models.resnet50(weights=None)  # No pre-trained weights
    num_classes = len(checkpoint['class_to_idx'])
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=num_classes)
    )

    model.load_state_dict(checkpoint['model_state_dict'])  # Use correct key for state dict

    return model

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    image_predictor = ImagePredictor(model, args.category_names)
    image_predictor.display_predictions(args.image_path, topk=args.top_k)

if __name__ == '__main__':
    main()

