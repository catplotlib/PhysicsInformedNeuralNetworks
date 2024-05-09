import torch

def generate_grid_points(x_min, x_max, t_min, t_max, nx, nt):
    x = torch.linspace(x_min, x_max, nx)
    t = torch.linspace(t_min, t_max, nt)
    x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
    return x_grid, t_grid

def generate_initial_condition(x, ic_function):
    return ic_function(x)

def generate_boundary_condition(t, bc_function):
    return bc_function(t)

def generate_collocation_points(x_min, x_max, t_min, t_max, n_points):
    x = torch.rand(n_points, 1) * (x_max - x_min) + x_min
    t = torch.rand(n_points, 1) * (t_max - t_min) + t_min
    return x, t

def generate_training_data(x_min, x_max, t_min, t_max, nx, nt, ic_function, bc_function):
    x_grid, t_grid = generate_grid_points(x_min, x_max, t_min, t_max, nx, nt)
    
    x_ic = x_grid[0]
    u_ic = generate_initial_condition(x_ic, ic_function)
    
    x_bc = x_grid[:, [0, -1]]
    t_bc = t_grid[[0, -1], :]
    u_bc = generate_boundary_condition(t_bc, bc_function)
    
    x_collocation, t_collocation = generate_collocation_points(
        x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max(), n_points=1000
    )
    
    return x_ic, t_bc, x_bc, u_ic, u_bc, x_collocation, t_collocation