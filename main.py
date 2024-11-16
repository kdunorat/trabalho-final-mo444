import torch
from process_data import process_data
from train_val_loader import generate_train_val
from model import model_build
from train_model import train, plot_training_history, evaluate


def generate_data(input_folder, output_folder, extract_faces=False):
    if extract_faces:
        process_data(input_folder, output_folder)
    print("Loading data...")
    train_loader, validation_loader = generate_train_val(output_folder)

    return train_loader, validation_loader

def model_results(train_loader, validation_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_build(device)
    print("Training model...")
    model_trained, history = train(device, train_loader, validation_loader, model, epochs=20)
    print(history)
    plot_training_history(history, name_graph='20_epocas')
    evaluate(device, model_trained, validation_loader)

if __name__ == '__main__':
    input_folder = './Dataset'
    output_folder = '/home/kdunorat/projetos/dados/processed_faces'
    train_loader, validation_loader =  generate_data(input_folder, output_folder)
    model_results(train_loader, validation_loader)