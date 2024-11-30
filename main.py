import torch
from model import model_build
from train_model import train, plot_training_history
from utils import load_model, generate_train_val, evaluate, plot_roc_auc
from sklearn.metrics import auc


def evaluation(model, validation_loader, name_graph=None, history=None):
    if history:
        if not name_graph:
            name_graph = 'Loss and accuracy by epochs'
        plot_training_history(history, name_graph=name_graph)
    
    evaluate(device, model, validation_loader)


if __name__ == '__main__':
    # input_folder = ''
    nome_graph = '20_epocas-128n-noMTCNN'
    output_folder = '/home/kdunorat/projetos/dados/processed_faces'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader =  generate_train_val(output_folder, batch=32)

    # Descongelando a ultima camada
    model_partial_defrost = model_build(device, dense_neurons=128, defrost=True, defrost_layers=1)

    # Training a model
    model_trained, history = train(device, train_loader, validation_loader, model_partial_defrost, epochs=20)
    evaluation(model_trained, validation_loader, name_graph=nome_graph, history=history)

    # Loading a model .pth
    # model_loaded = load_model(model, '20epochs-128n-checkpoint.pth')

    # Getting metrics
    fpr, tpr = evaluate(device='cuda', model=model_trained, validation_loader=validation_loader)
    # Save auc
    plot_roc_auc(fpr, tpr, auc(fpr, tpr), name_graph=f'{nome_graph}_roc-auc')
   