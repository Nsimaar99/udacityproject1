import argparse
import torch
import os
from data import setup_flower_data
from dataset import FlowerDataset
from model import FlowerResNet50

def get_input_args():
    """
    Parses command line arguments for training the model.
    """
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset and save the model as a checkpoint.")

    # Basic command-line arguments
    parser.add_argument('data_dir', type=str, help='Path to the dataset folder')

    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model checkpoint')
    parser.add_argument('--arch', type=str, default='resnet50', help='Model architecture: vgg13, resnet50, etc.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    return parser.parse_args()

def train_and_validate(model, criterion, optimizer, dataloaders, device, epochs=5, print_every=20, checkpoint_path='checkpoint.pth'):
    """
    Trains and validates the model, saving checkpoints during the process.
    """
    steps = 0

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.type(torch.long).to(device)

            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()  # Set model to evaluation mode
                test_loss = 0
                accuracy = 0

                with torch.inference_mode():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.type(torch.long).to(device)

                        # Forward pass
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                        test_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.softmax(logits, dim=-1)
                        top_class = ps.argmax(dim=-1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()  # Switch back to training mode

        # Save a checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': dataloaders['train'].dataset.class_to_idx,
            'num_classes': model.fc[-1].out_features,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    return model

if __name__ == "__main__":
    # Get command line arguments
    args = get_input_args()

    # Check device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Step 1: Load data
    setup_flower_data()
    flower_dataset = FlowerDataset(args.data_dir, batch_size=64)
    train_loader, valid_loader, test_loader = flower_dataset.get_dataloaders()
    dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    # Step 2: Initialize the model (default to ResNet50 if no architecture is specified)
    custom_resnet = CustomResNet50(num_classes=102, learning_rate=args.learning_rate)
    model = custom_resnet.get_model()
    criterion = custom_resnet.get_criterion()
    optimizer = custom_resnet.get_optimizer()

    # Step 3: Train and validate the model, and save checkpoints
    trained_model = train_and_validate(model, criterion, optimizer, dataloaders, device,
                                       epochs=args.epochs, print_every=20, checkpoint_path=os.path.join(args.save_dir, 'checkpoint.pth'))
