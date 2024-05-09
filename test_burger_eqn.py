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
nu = 0.01  # Viscosity

# Initial and boundary conditions for Burgers' equation
def ic_function(x):
    return -torch.sin(torch.pi * x)

def bc_function(t):
    return torch.zeros_like(t)

# PDE function for Burgers' equation
def pde_function(u, u_t, u_x, u_xx):
    return u_t + u * u_x - nu * u_xx

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
num_epochs = 100
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
    input_batch = torch.cat([x_collocation_batch, x_ic_batch, x_bc_batch], dim=0)
    t_batch = torch.cat([t_collocation_batch, torch.zeros_like(x_ic_batch), t_bc_batch], dim=0)
    u_batch = torch.cat([torch.zeros_like(x_collocation_batch), u_ic_batch, u_bc_batch], dim=0)

    # Forward pass
    u_pred = model(torch.cat([input_batch, t_batch], dim=-1))

    # Compute the loss
    loss = physics_informed_loss(u_pred, u_batch, model, input_batch, t_batch, pde_function)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for monitoring
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on test data
with torch.no_grad():
    x_test = torch.linspace(x_min, x_max, 100)
    t_test = torch.linspace(t_min, t_max, 100)
    x_test_grid, t_test_grid = torch.meshgrid(x_test, t_test, indexing='ij')
    input_test = torch.stack([x_test_grid.reshape(-1), t_test_grid.reshape(-1)], dim=1)
    u_test = model(input_test).reshape(x_test_grid.shape)

# Visualize the solution
plt.figure(figsize=(8, 6))
plt.pcolormesh(x_test_grid.detach().numpy(), t_test_grid.detach().numpy(), u_test.detach().numpy(), cmap='viridis', shading='auto')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Solution to Burgers\' Equation')
plt.tight_layout()
plt.show()

def exact_solution(x, t, nu=0.01):
    # Ensure x and t are in tensor format and can broadcast against each other
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    
    if len(x.shape) == 1:
        x = x.unsqueeze(1)  # Make x a column vector if it's a 1D tensor
    if len(t.shape) == 1:
        t = t.unsqueeze(0)  # Make t a row vector if it's a 1D tensor

    # Burgers' equation analytical solution under specific initial and boundary conditions
    u_exact = -torch.sin(torch.pi * x) / (1 + torch.pi * t * torch.cos(torch.pi * x))

    return u_exact

# Assuming x_test and t_test are generated as 1D tensors
x_test_grid, t_test_grid = torch.meshgrid(x_test, t_test, indexing='ij')

# Assuming an exact solution function exact_solution(x, t) is defined
u_exact = exact_solution(x_test_grid, t_test_grid).reshape(x_test_grid.shape)


fig = go.Figure(data=[
    go.Surface(z=u_test.detach().numpy(), x=x_test_grid.detach().numpy(), y=t_test_grid.detach().numpy(), colorscale='Viridis', name='Predicted'),
    go.Surface(z=u_exact.detach().numpy(), x=x_test_grid.detach().numpy(), y=t_test_grid.detach().numpy(), opacity=0.9, name='Exact')
])

fig.update_layout(title='Predicted vs Exact Solution', autosize=False,
                  width=700, height=700,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()


specific_x_indices = [25, 50, 75]  # Example indices for x = 0.25, 0.5, 0.75

plt.figure(figsize=(10, 7))
for idx in specific_x_indices:
    plt.plot(t_test.detach().numpy(), u_test[:, idx].detach().numpy(), label=f'x={x_test[idx].item():.2f}')

plt.xlabel('Time (t)')
plt.ylabel('u(x,t)')
plt.title('Time Series of Predicted Solutions at Different x')
plt.legend()
plt.grid(True)
plt.show()
