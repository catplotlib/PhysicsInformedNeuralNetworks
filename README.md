# Physics-Informed Neural Networks for Solving PDEs and ODEs

This repository contains an implementation of Physics-Informed Neural Networks (PINNs) for solving partial differential equations (PDEs) and ordinary differential equations (ODEs). PINNs leverage the power of neural networks to approximate the solution of PDEs and ODEs by incorporating the governing equations and boundary conditions into the loss function.

## Features

- Solve PDEs and ODEs using physics-informed neural networks
- Flexible and modular framework for defining custom PDEs, ODEs, initial conditions, and boundary conditions
- Easy-to-use training and evaluation pipeline
- Visualization of the approximated solution

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/physics-informed-neural-networks.git
   ```

2. Install the required dependencies:
   ```
   pip install torch numpy matplotlib
   ```

## Usage

1. Define your problem:
   - Specify the PDE or ODE governing equation in the `pde_function` or `ode_function`.
   - Define the initial and boundary conditions using the `ic_function` and `bc_function`.
   - Set the problem domain and parameters, such as the spatial and temporal ranges and the number of training points.

2. Create an instance of the `PhysicsInformedNN` model:
   ```python
   model = PhysicsInformedNN(input_dim, hidden_dim, output_dim)
   ```

3. Generate the training data using the `generate_training_data` function:
   ```python
   x_ic, t_bc, x_bc, u_ic, u_bc, x_collocation, t_collocation = generate_training_data(...)
   ```

4. Create an optimizer and train the model:
   ```python
   optimizer = get_optimizer(model)
   for epoch in range(num_epochs):
       # Training loop
       ...
   ```

5. Evaluate the trained model on test data and visualize the approximated solution:
   ```python
   with torch.no_grad():
       # Evaluate the model on test data
       ...
   
   # Visualize the solution
   plt.figure(figsize=(8, 6))
   plt.pcolormesh(...)
   plt.colorbar(label='Solution')
   plt.xlabel('x')
   plt.ylabel('t')
   plt.title('PDE/ODE Solution')
   plt.tight_layout()
   plt.show()
   ```

## Examples

The repository includes example scripts for solving different PDEs and ODEs:

- `test_heat_equation.py`: Solves the 1D heat equation, a parabolic PDE.
- `test_burger_equation.py`: Solves the 1D Burger's equation, a parabolic PDE.

You can use these examples as a starting point and modify them to solve your specific PDEs or ODEs.

## Code Structure

- `physicsinformed/models/physics_informed_models.py`: Contains the implementation of the Physics-Informed Neural Network (PINN) model.

- `physicsinformed/models/losses.py`: Defines the physics-informed loss functions for PDEs and ODEs.

- `physicsinformed/training/optimizers.py`: Provides the optimizer used for training the PINN model.

- `physicsinformed/utils/data_utils.py`: Contains utility functions for generating training data.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments

The implementation of Physics-Informed Neural Networks in this repository is based on the research paper:

- M. Raissi, P. Perdikaris, and G. E. Karniadakis. "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations." Journal of Computational Physics, 2019.

Special thanks to the authors for their groundbreaking work in this field.