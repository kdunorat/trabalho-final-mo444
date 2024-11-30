import torch
from model import model_build
from train_model import train, plot_training_history
from utils import load_model, process_data, generate_train_val, evaluate, plot_roc_auc
from sklearn.metrics import auc


def generate_data(output_folder, extract_faces=False, input_folder=None):
    if extract_faces:
        process_data(input_folder, output_folder)
    print("Loading data...")
    train_loader, validation_loader = generate_train_val(output_folder)

    return train_loader, validation_loader


def evaluation(model, validation_loader, name_graph=None, history=None):
    if history:
        if not name_graph:
            name_graph = 'Loss and accuracy by epochs'
        plot_training_history(history, name_graph=name_graph)
    
    evaluate(device, model, validation_loader)


if __name__ == '__main__':
    # input_folder = ''
    # output_folder = '/home/kdunorat/projetos/dados/processed_faces'
    output_folder = '/home/kdunorat/projetos/dados/Dataset' # Tentando sem o mtcnn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader =  generate_data(output_folder=output_folder)

    # Descongelando a ultima camada
    model_partial_defrost = model_build(device, dense_neurons=128, defrost=True, defrost_layers=1)

    # Training a model
    model_trained, history = train(device, train_loader, validation_loader, model_partial_defrost, epochs=20)
    evaluation(model_trained, validation_loader, name_graph='20_epocas-128n-noMTCNN', history=history)

    # Loading a model .pth
    # model_loaded = load_model(model, '20epochs-128n-checkpoint.pth')
    # Getting metrics
    fpr, tpr = evaluate(device='cuda', model=model_trained, validation_loader=validation_loader)
    # Save auc
    plot_roc_auc(fpr, tpr, auc(fpr, tpr), name_graph='roc-auc-20epochs-noMTCNN')
   