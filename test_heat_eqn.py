import torch
from physicsinformed.models.physics_informed_models import PhysicsInformedNN
from physicsinformed.models.losses import physics_informed_loss
from physicsinformed.training.optimizers import get_optimizer
from physicsinformed.utils.data_utils import generate_training_data
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Define the problem domain and parameters
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0
nx, nt = 100, 100

# Define the initial and boundary condition functions
def ic_function(x):
    return torch.sin(2 * torch.pi * x)

def bc_function(t):
    return torch.zeros_like(t)

# Define the PDE function
def pde_function(u, u_t, u_x, u_xx):
    return u_t - u_xx

# Generate the training data
x_ic, t_bc, x_bc, u_ic, u_bc, x_collocation, t_collocation = generate_training_data(
    x_min, x_max, t_min, t_max, nx, nt, ic_function, bc_function
)

# Create the physics-informed model
input_dim = 2
hidden_dim = 32
output_dim = 1
model = PhysicsInformedNN(input_dim, hidden_dim, output_dim)

# Create the optimizer
optimizer = get_optimizer(model)

# Training loop
num_epochs = 1000
batch_size = 32

for epoch in range(num_epochs):
    # Shuffle and batch the training data
    indices = torch.randperm(x_collocation.shape[0])
    x_collocation_batch = x_collocation[indices][:batch_size]
    t_collocation_batch = t_collocation[indices][:batch_size]
    x_ic_batch = x_ic.reshape(-1, 1)
    u_ic_batch = u_ic.reshape(-1, 1)
    x_bc_batch = x_bc.reshape(-1, 1)
    t_bc_batch = t_bc.reshape(-1, 1)
    u_bc_batch = u_bc.reshape(-1, 1)

    # Concatenate the input data along the last dimension
    x_batch = torch.cat([x_collocation_batch, x_ic_batch, x_bc_batch], dim=0)
    t_batch = torch.cat([t_collocation_batch, torch.zeros_like(x_ic_batch), t_bc_batch], dim=0)
    input_batch = torch.cat([x_batch, t_batch], dim=-1)
    u_batch = torch.cat([torch.zeros_like(x_collocation_batch), u_ic_batch, u_bc_batch], dim=0)
    # Forward pass
    u_pred = model(input_batch)

    # Compute the loss
    loss = physics_informed_loss(u_pred, u_batch, model, x_batch, t_batch, pde_function)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on test data
with torch.no_grad():
    x_test = torch.linspace(x_min, x_max, 100)
    t_test = torch.linspace(t_min, t_max, 100)
    x_test_grid, t_test_grid = torch.meshgrid(x_test, t_test, indexing='ij')
    x_test_flat = x_test_grid.reshape(-1)
    t_test_flat = t_test_grid.reshape(-1)
    u_test = model(torch.stack([x_test_flat, t_test_flat], dim=1)).reshape(x_test_grid.shape)

# Visualize the solution
plt.figure(figsize=(8, 6))
plt.pcolormesh(x_test_grid.detach().numpy(), t_test_grid.detach().numpy(), u_test.detach().numpy(), cmap='viridis', shading='auto')
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Heat Equation Solution')
plt.tight_layout()
plt.show()

