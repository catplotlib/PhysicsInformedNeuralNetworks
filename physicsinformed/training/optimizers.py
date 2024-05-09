import torch.optim as optim

def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)