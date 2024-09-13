import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json

# Load the checkpoint and rebuild the model
def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load the ResNet50 model
    model = models.resnet50(weights=None)
    num_classes = len(checkpoint['class_to_idx'])
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Return the model and the class_to_idx dictionary
    return model, checkpoint['class_to_idx']

class ImagePredictor:
    def __init__(self, model, class_to_idx, device, json_file_path='cat_to_name.json'):
        self.model = model
        self.class_to_idx = class_to_idx
        self.device = device
        # Reverse the class_to_idx to get idx_to_class
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.label_mapping = self.load_label_mapping(json_file_path)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Load the category names from a JSON file
    def load_label_mapping(self, json_file_path):
        with open(json_file_path, 'r') as f:
            return json.load(f)

    # Get flower name from the class index
    def get_flower_name(self, class_index):
        # Map class index to the original class
        original_class_index = self.idx_to_class.get(class_index, "Unknown")
        return self.label_mapping.get(str(original_class_index), "Unknown class")

    # Process the input image
    def process_image(self, image_path):
        image = Image.open(image_path)
        return self.preprocess(image).unsqueeze(0).to(self.device)

    # Predict the class of the image
    def predict(self, image_path, topk=5):
        self.model.eval()
        image_tensor = self.process_image(image_path)
        with torch.no_grad():
            output = self.model(image_tensor)
            ps = torch.softmax(output, dim=1)
            top_p, top_indices = ps.topk(topk, dim=1)

        # Convert indices to actual class labels
        top_p = top_p.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        
        top_classes = [self.idx_to_class[idx] for idx in top_indices]
        top_flowers = [self.get_flower_name(idx) for idx in top_indices]

        return top_p, top_flowers

    # Display the predictions
    def display_predictions(self, image_path, topk=5):
        top_p, top_flowers = self.predict(image_path, topk)
        print(f"Top {topk} predictions for the image:")
        for i, (flower, prob) in enumerate(zip(top_flowers, top_p)):
            print(f"{i + 1}: {flower} with probability {prob:.4f}")

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the category names file.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')
    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    # Determine the device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load the model from the checkpoint
    model, class_to_idx = load_checkpoint(args.checkpoint)
    model.to(device)

    # Create ImagePredictor instance and display predictions
    image_predictor = ImagePredictor(model, class_to_idx, device, args.category_names)
    image_predictor.display_predictions(args.image_path, topk=args.top_k)

if __name__ == '__main__':
    main()
