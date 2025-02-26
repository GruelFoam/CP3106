import torch
from model.ae import AE
from model.clasf import ClassificationModel

def load_model(model_class, device, task, model_path=None, *model_args, **model_kwargs):
    """
    Load a PyTorch model from a .pth file.

    Parameters:
    - model_path (str): Path to the saved model file.
    - model_class (torch.nn.Module): The model class to instantiate.
    - *model_args, **model_kwargs: Arguments to initialize the model before loading weights.

    Returns:
    - model (torch.nn.Module): The loaded model.
    """
    # Initialize the model architecture
    model = model_class(*model_args, **model_kwargs).to(device)
    
    if task == "load":
        # Load the saved state_dict
        model.load_state_dict(torch.load(model_path))
        model.eval()
    
    return model

# Autoencoder
def init_ae(device, input_dim, hidden_dim, latent_dim, drop_rate):
    '''
    Obtain an untrained ae
    '''
    return load_model(AE, device, task="train", input_dim=input_dim, hidden_dim=hidden_dim,
                      latent_dim=latent_dim, drop_rate=drop_rate)

def load_ae(model_path, device):
    '''
    Load the trained ae
    '''
    # Optimized hyper paramenters
    data_dim = 1536
    hidden_dim = 512
    latent_dim = 256
    drop_rate = 0.15
    return load_model(AE, device, task="load", model_path=model_path, input_dim=data_dim, hidden_dim=hidden_dim,
                      latent_dim=latent_dim, drop_rate=drop_rate)

# Classification model
def init_clasf(device, input_dim, hidden_dim, num_classes, drop_rate):
    return load_model(ClassificationModel, device, task="train", input_dim=input_dim, hidden_dim=hidden_dim,
                      num_classes=num_classes, drop_rate=drop_rate)

def load_clasf(model_path, device):
    '''
    Load the trained ae
    '''
    # Optimized hyper paramenters
    input_dim = 256
    hidden_dim = 64
    num_classes = 11
    drop_rate = 0.1
    return load_model(AE, device, task="load", model_path=model_path, input_dim=input_dim, hidden_dim=hidden_dim,
                      num_classes=num_classes, drop_rate=drop_rate)