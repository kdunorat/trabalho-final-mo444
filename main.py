import torch
from model import model_build
from train_model import train, plot_training_history
from utils import load_model, generate_train_val, evaluate, plot_roc_auc
from sklearn.metrics import auc
from torchvision import models
import torch.nn as nn


def evaluation(model, validation_loader, name_graph=None, history=None):
    if history:
        if not name_graph:
            name_graph = 'Loss and accuracy by epochs'
        plot_training_history(history, name_graph=name_graph)
    evaluate(device, model, validation_loader)
    


if __name__ == '__main__':
    # Mudar isso:
    nome_graph = 'defrost-3-gabriel'
    input_folder = '/home/kdunorat/projetos/dados/processed_faces'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader =  generate_train_val(input_folder, batch=32)

    # Descongelando a ultima camada
    # model = model_build(device)
    model_partial_defrost = model_build(device, dense_neurons=128, defrost=True, defrost_layers=3)

    # Training a model
    # model_trained, history = train(device, train_loader, validation_loader, model_partial_defrost, epochs=20)
    # plot_training_history(history, name_graph=nome_graph)
    
    # Loading a model .pth
    model_loaded = load_model(model_partial_defrost, 'model_3_defrost.pth')

    # Getting metrics
    fpr, tpr = evaluate(device='cuda', model=model_loaded, validation_loader=test_loader)
    # Save auc
    plot_roc_auc(fpr, tpr, auc(fpr, tpr), name_graph=f'{nome_graph}_roc-auc')
   