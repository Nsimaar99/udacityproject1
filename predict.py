import argparse
import torch
from PIL import Image
import json
from torchvision import models, transforms
from collections import OrderedDict

# Define the ImagePredictor class as described earlier
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
        size = 256
        crop_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        image = image.resize((size, size))
        left = (size - crop_size) / 2
        top = (size - crop_size) / 2
        right = (size + crop_size) / 2
        bottom = (size + crop_size) / 2
        image = image.crop((left, top, right, bottom))
        image_array = np.array(image) / 255.0
        image_array = (image_array - mean) / std
        image_array = image_array.transpose((2, 0, 1))
        return image_array

    def predict(self, image_path, topk=5):
        image = Image.open(image_path)
        image_array = self.process_image(image)
        image_tensor = torch.tensor(image_array).float().unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, topk)
            top_probs = top_probs.numpy().flatten()
            top_indices = top_indices.numpy().flatten()
        return top_probs, top_indices

    def display_predictions(self, image_path, topk=5):
        top_probs, top_indices = self.predict(image_path, topk)
        print(f"Top {topk} Predictions:")
        for i in range(len(top_probs)):
            class_name = self.get_flower_name(top_indices[i])
            print(f"Flower name: {class_name}, Probability: {top_probs[i]}")

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['state_dict'])
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
    model = load_model(args.checkpoint)
    model.to(device)

    image_predictor = ImagePredictor(model, args.category_names)
    image_predictor.display_predictions(args.image_path, topk=args.top_k)

if __name__ == '__main__':
    main()
