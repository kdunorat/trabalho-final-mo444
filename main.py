import torch
from process_data import process_data
from train_val_loader import generate_train_val
from model import model_build
from train_model import train, plot_training_history, evaluate
from utils import load_model


def generate_data(input_folder, output_folder, extract_faces=False):
    if extract_faces:
        process_data(input_folder, output_folder)
    print("Loading data...")
    train_loader, validation_loader = generate_train_val(output_folder)

    return train_loader, validation_loader


def model_results(train_loader, validation_loader):
    print("Training model...")
    model_trained, history = train(device, train_loader, validation_loader, model, epochs=20)
    return model_trained, history


def evaluation(model, validation_loader, name_graph=None, history=None):
    if history:
        if not name_graph:
            name_graph = 'Loss and accuracy by epochs'
        plot_training_history(history, name_graph=name_graph)
    
    evaluate(device, model, validation_loader)


if __name__ == '__main__':
    input_folder = './Dataset'
    output_folder = '/home/kdunorat/projetos/dados/processed_faces'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader =  generate_data(input_folder, output_folder)
    # Using 128 neurons
    model = model_build(device)

    # Training a model
    # model_trained, history = model_results(train_loader, validation_loader)
    # evaluation(model_trained, validation_loader, name_graph='20_epocas-128n', history=history)

    # Loading a model 
    model_loaded = load_model(model, 'best_model.pth')
    evaluation(model_loaded, validation_loader, name_graph='20_epocas-128n')
   
