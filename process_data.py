import os
from PIL import Image
from face_extract import extract_face

def process_data(input_folder: str, output_folder: str):
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

