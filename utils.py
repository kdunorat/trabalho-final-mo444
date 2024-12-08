import os
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score, precision_score, roc_curve


def load_model(model, pth:str):
    print("\nLoading model...")
    model.load_state_dict(torch.load(pth))
    model.eval()
    return model

def preprocess_image(image_path: str, transform, device):
    image = Image.open(image_path).convert("RGB")  # Carregar como RGB
    image_tensor = transform(image).unsqueeze(0).to(device)  # Aplicar transformações e adicionar dimensão do batch
    return image_tensor

def generate_train_val(output_folder: str, batch: int):
    torch.cuda.empty_cache()
    # Data augmentation and normalization for training
    data_transforms = {
        'Train': transforms.Compose([
            # Data augmentation:
            #transforms.Resize((256, 256)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(output_folder, 'Train'), data_transforms['Train'])
    validation_dataset = datasets.ImageFolder(os.path.join(output_folder, 'Validation'), data_transforms['Validation'])
    test_dataset = datasets.ImageFolder(os.path.join(output_folder, 'Test'), data_transforms['Test'])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)


    return train_loader, validation_loader, test_loader


def evaluate_and_get_tp_tn(device, model, validation_loader):
    print("\nEvaluating Model...")
    all_labels = []
    all_predictions = []
    y_scores = []
    correct, total = 0, 0

    true_positive = None
    true_negative = None

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            predicted = (outputs >= 0.5).float()
            y_scores.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

           # Collect TP and TN examples
            for i in range(len(labels)):
                if labels[i] == 1 and predicted[i] == 1 and true_positive is None:  # True Positive
                    true_positive = inputs[i].unsqueeze(0)  # Add batch dimension
                if labels[i] == 0 and predicted[i] == 0 and true_negative is None:  # True Negative
                    true_negative = inputs[i].unsqueeze(0)  # Add batch dimension
                if true_positive is not None and true_negative is not None:
                    break

    val_accuracy = correct / total
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)

    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, y_scores)
    
    return fpr, tpr, true_positive, true_negative



def plot_roc_auc(fpr, tpr, roc_auc, name_graph):
    # Plotting the ROC AUC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'{name_graph}.png')


def generate_gradcam(model, target_layer, input_image, target_class=None):
    # Register hooks to access gradients and activations
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_image).squeeze(-1)  # Ensure output is 1D
    if target_class is None:
        target_class = torch.argmax(output).item()  # Default to the highest class
    loss = output[0]  # Use the single output value

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Extract gradients and activations
    gradient = gradients[0].cpu().detach().numpy()[0]
    activation = activations[0].cpu().detach().numpy()[0]

    # Compute weights
    weights = np.mean(gradient, axis=(1, 2))
    gradcam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        gradcam += w * activation[i]

    gradcam = np.maximum(gradcam, 0)
    gradcam = cv2.resize(gradcam, (input_image.shape[3], input_image.shape[2]))
    gradcam = gradcam - np.min(gradcam)
    gradcam = gradcam / np.max(gradcam)
    
    return gradcam

# Visualizing Grad-CAM
def visualize_gradcam(input_image, gradcam, name_graph):
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    input_image = input_image.cpu().numpy().transpose(1, 2, 0)
    cam = heatmap + np.float32(input_image)
    cam = cam / np.max(cam)
    plt.imshow(cam)
    plt.axis('off')
    plt.savefig(f'{name_graph}.png')
