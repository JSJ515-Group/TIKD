import torch
import argparse
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from teacher_models.resnet import resnet32x4  # Import resnet32x4 function

# Create parser
parser = argparse.ArgumentParser(description='Evaluation Script for ResNet32x4')
parser.add_argument('--teacher_path', type=str, required=True, help='Path to the teacher model checkpoint')
args = parser.parse_args()

# Set random seed
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = torchvision.datasets.CIFAR100(root="./dataset/cifar-100-python", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
teacher_model = resnet32x4(num_classes=100)  # Set num_classes for CIFAR-100
teacher_model = teacher_model.to(device)

# Load saved checkpoint
checkpoint = torch.load(args.teacher_path, map_location=device)
teacher_model.load_state_dict(checkpoint['model'])  # Load only the model parameters
best_acc = checkpoint['best_acc']  # Get the saved best accuracy
teacher_model.eval()

# Function to calculate accuracy
def calculate_accuracy(loader, model, topk=(1, 5)):
    model.eval()  # Set to evaluation mode
    correct = [0] * len(topk)
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.topk(max(topk), 1, True, True)
            predicted = predicted.t()
            correct_cases = predicted.eq(labels.view(1, -1).expand_as(predicted))
            for i, k in enumerate(topk):
                correct[i] += correct_cases[:k].reshape(-1).float().sum(0, keepdim=True).item()
            total += labels.size(0)

    return [correct[i] / total * 100 for i in range(len(topk))]

# Calculate Top-1 and Top-5 accuracy of the model
acc1, acc5 = calculate_accuracy(test_loader, teacher_model)
print(f'Accuracy Top-1: {acc1:.2f}%, Accuracy Top-5: {acc5:.2f}%')
print(f'Best Accuracy (Top-1 during training): {best_acc:.2f}%')