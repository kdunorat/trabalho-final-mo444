import torch
from facenet_pytorch import MTCNN
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN with device for CUDA support
detector = MTCNN(keep_all=False, device=device)

def extract_face(filename, required_size=(256, 256)):
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
