import torch
from model import model_build
from train_model import train, plot_training_history
from utils import load_model, generate_train_val, evaluate_and_get_tp_tn, plot_roc_auc, preprocess_image, generate_gradcam, visualize_gradcam
from sklearn.metrics import auc
from torchvision import models
import torch.nn as nn
from torchvision.transforms import transforms


if __name__ == '__main__':
    # Mudar isso:
    nome_graph = 'defrost-3-gabriel'
    input_folder = '/home/kdunorat/projetos/dados/processed_faces'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader =  generate_train_val(input_folder, batch=4, small_test=True) # Diminuí pra testar o gradcam.

    # Descongelando a ultima camada
    # model = model_build(device)
    model_partial_defrost = model_build(device, dense_neurons=128, defrost=True, defrost_layers=2)

    # Training a model
    # model_trained, history = train(device, train_loader, validation_loader, model_partial_defrost, epochs=20)
    # plot_training_history(history, name_graph=nome_graph)
    
    # Loading a model .pth
    model_loaded = load_model(model_partial_defrost, 'model_2_defrost.pth')

    # Getting metrics
    # fpr, tpr, _, _ = evaluate_and_get_tp_tn(device='cuda', model=model_loaded, validation_loader=test_loader)
    # Save auc
    # plot_roc_auc(fpr, tpr, auc(fpr, tpr), name_graph=f'{nome_graph}_roc-auc')

    ###################################### Generate Grad-Cam

    target_layer = model_loaded.features[-1]  # Last convolutional layer
    _, _, real_prediction, false_prediction = evaluate_and_get_tp_tn(device, model_loaded, test_loader)


    # Gerar Grad-CAM para cada classe
    gradcam_real = generate_gradcam(model_loaded, target_layer, real_prediction, target_class=0)  # Classe "Real"
    gradcam_fake = generate_gradcam(model_loaded, target_layer, false_prediction, target_class=1)  # Classe "Fake"

    # Visualizar os mapas de Grad-CAM
    visualize_gradcam(real_prediction.squeeze(0), gradcam_real, name_graph='Grad-cam_defrost_2-real')
    visualize_gradcam(false_prediction.squeeze(0), gradcam_fake, name_graph='Grad-cam_defrost_2-fake')

   