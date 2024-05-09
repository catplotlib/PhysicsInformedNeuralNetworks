import torch

def mean_squared_error_with_constraints(output, target, constraints):
    mse_loss = torch.mean((output - target) ** 2)
    constraint_loss = torch.mean(torch.sum(constraints ** 2, dim=1))
    return mse_loss + constraint_loss

def physics_informed_loss(output, target, model, x, t, pde_function):
    data_loss = torch.mean((output - target) ** 2)
    pde_residual = model.compute_pde_residual(x, t, pde_function)
    pde_loss = torch.mean(pde_residual ** 2)
    return data_loss + pde_loss