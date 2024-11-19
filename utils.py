import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_model(model, pth:str):
    model.load_state_dict(torch.load(pth))
    model.eval()
    return model


def process_data(device, input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    print("Starting face extraction")
    for split in ['Train', 'Validation', 'Test']:
        split_folder = os.path.join(input_folder, split)
        output_split_folder = os.path.join(output_folder, split)
        os.makedirs(output_split_folder, exist_ok=True)
        
        for label in ['Real', 'Fake']:
            label_folder = os.path.join(split_folder, label)
            output_label_folder = os.path.join(output_split_folder, label)
            os.makedirs(output_label_folder, exist_ok=True)
            
            for filename in os.listdir(label_folder):
                filepath = os.path.join(label_folder, filename)
                face = extract_face(filepath)
                if face is not None:
                    output_path = os.path.join(output_label_folder, filename)
                    face.save(output_path, format='JPEG')


def extract_face(device, filename, required_size=(256, 256)):
    # Initialize MTCNN with device for CUDA support
    detector = MTCNN(keep_all=False, device=device)
    # Load image using PIL and convert to RGB
    image = Image.open(filename).convert('RGB')
    
    # Detect face and extract it
    boxes, _ = detector.detect(image)
    
    # If a face is detected
    if boxes is not None:
        x, y, x2, y2 = [int(coord) for coord in boxes[0]]
        # Extract the face from the image
        face = image.crop((x, y, x2, y2))
        # Resize the face to the required size
        face = face.resize(required_size)
        return face
    
    return None


def generate_train_val(output_folder: str):
    torch.cuda.empty_cache()
    # Data augmentation and normalization for training
    data_transforms = {
        'Train': transforms.Compose([
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
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(output_folder, 'Train'), data_transforms['Train'])
    validation_dataset = datasets.ImageFolder(os.path.join(output_folder, 'Validation'), data_transforms['Validation'])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)


    return train_loader, validation_loader