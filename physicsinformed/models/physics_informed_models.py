import torch
import torch.nn as nn
from .custom_layers import SineLayer

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PhysicsInformedNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)  
        self.hidden_layer = SineLayer(hidden_dim, hidden_dim) 
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.sin(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def compute_pde_residual(self, x, t, pde_function):
        x.requires_grad_(True)
        t.requires_grad_(True)
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        combined_input = torch.cat([x, t], dim=1)
        u = self.forward(combined_input)

        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        residual = pde_function(u, u_t, u_x, u_xx)
        return residual