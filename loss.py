import torch
import random
'''
linear elasticity constraint registration model

'''
def mae_loss(x, y):
    return 0.1*torch.mean((x - y) ** 2)

def chamfer_loss(x, y, ps):
    A = x 
    B = y  
    r = torch.sum(A * A, dim=2).unsqueeze(-1) 
    r1 = torch.sum(B * B, dim=2).unsqueeze(-1)
    t = r.repeat(1, 1, ps) - 2 * torch.bmm(A, B.permute(0, 2, 1)) + r1.permute(0, 2, 1).repeat(1, ps, 1)
    d1, _ = t.min(dim=2)
    d2, _ = t.min(dim=1)
    unsquared_d1 = d1
    unsquared_d2 = d2   
    sum_d1 = unsquared_d1.sum(dim=1)
    sum_d2 = unsquared_d2.sum(dim=1)
    chamfer_distance = 0.5 * (sum_d1 / ps + sum_d2 / ps)
    return chamfer_distance.mean()


def linear_elastic_loss(data, displacements, strain_source, yms, prs):
    strain_source = strain_source.permute(0, 2, 1)

    upsilon = prs
    E = yms

    G = E / (2 * (1 + upsilon))
    K = E / (3 * (1 - 2 * upsilon))
    lambda_ = (E * upsilon) / ((1 + upsilon) * (1 - 2 * upsilon))

    u = displacements[:, 0, :]
    v = displacements[:, 1, :]
    w = displacements[:, 2, :]

    grad_outputs = torch.ones_like(u)
    du_ddata = torch.autograd.grad(
        outputs=u,
        inputs= data,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]

    grad_outputs = torch.ones_like(v)
    dv_ddata = torch.autograd.grad(
        outputs=v,
        inputs=data,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]

    grad_outputs = torch.ones_like(w)
    dw_ddata = torch.autograd.grad(
        outputs=w,
        inputs=data,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]

    eps_du_dx = du_ddata[:, 0, :, 0]  # ∂u/∂x
    eps_du_dy = du_ddata[:, 0, :, 1]  # ∂u/∂y
    eps_du_dz = du_ddata[:, 0, :, 2]  # ∂u/∂z
    eps_dv_dx = dv_ddata[:, 0, :, 0]  # ∂v/∂x
    eps_dv_dy = dv_ddata[:, 0, :, 1]  # ∂v/∂y
    eps_dv_dz = dv_ddata[:, 0, :, 2]  # ∂v/∂z
    eps_dw_dx = dw_ddata[:, 0, :, 0]  # ∂w/∂x
    eps_dw_dy = dw_ddata[:, 0, :, 1]  # ∂w/∂y
    eps_dw_dz = dw_ddata[:, 0, :, 2]  # ∂w/∂z

    residual_strain = torch.square(torch.mean(
                            ((strain_source[:, :, 0]-eps_du_dx)) +
                            ((strain_source[:, :, 1]-eps_dv_dy)) +
                            ((strain_source[:, :, 2]-eps_dw_dz)) +
                            ((strain_source[:, :, 3]-0.5*(eps_du_dy+eps_dv_dx))) +
                            ((strain_source[:, :, 4]-0.5*(eps_du_dz+eps_dw_dx))) +
                            ((strain_source[:, :, 5]-0.5*(eps_dv_dz+eps_dw_dy)))))




    sigma_xx = (lambda_ + 2 * G) * strain_source[:, :, 0]#
    sigma_yy = (lambda_ + 2 * G) * strain_source[:, :, 1]
    sigma_zz = (lambda_ + 2 * G) * strain_source[:, :, 2]
    sigma_xy = G * strain_source[:, :, 3] #0.5 * (du_dy + dv_dx)
    sigma_xz = G * strain_source[:, :, 4] #0.5 * (du_dz + dw_dx)
    sigma_yz = G * strain_source[:, :, 5] #0.5 * (dv_dz + dw_dy)

    sigma_xx_grad2 = torch.ones_like(sigma_xx)
    dsigma_xxddata = torch.autograd.grad(
        outputs=sigma_xx,
        inputs=data,
        grad_outputs=sigma_xx_grad2,
        create_graph=True,
        retain_graph=True
    )[0]
    sigma_yy_grad2 = torch.ones_like(sigma_yy)
    dsigma_yyddata = torch.autograd.grad(
        outputs=sigma_yy,
        inputs=data,
        grad_outputs=sigma_yy_grad2,
        create_graph=True,
        retain_graph=True
    )[0]
    sigma_zz_grad2 = torch.ones_like(sigma_zz)
    dsigma_zzddata = torch.autograd.grad(
        outputs=sigma_zz,
        inputs=data,
        grad_outputs=sigma_zz_grad2,
        create_graph=True,
        retain_graph=True
    )[0]
    sigma_xy_grad2 = torch.ones_like(sigma_xy)
    dsigma_xyddata = torch.autograd.grad(
        outputs=sigma_xy,
        inputs=data,
        grad_outputs=sigma_xy_grad2,
        create_graph=True,
        retain_graph=True
    )[0]
    sigma_xz_grad2 = torch.ones_like(sigma_xz)
    dsigma_xzddata = torch.autograd.grad(
        outputs=sigma_xz,
        inputs=data,
        grad_outputs=sigma_xz_grad2,
        create_graph=True,
        retain_graph=True
    )[0]
    sigma_yz_grad2 = torch.ones_like(sigma_yz)
    dsigma_yzddata = torch.autograd.grad(
        outputs=sigma_yz,
        inputs=data,
        grad_outputs=sigma_yz_grad2,
        create_graph=True,
        retain_graph=True
    )[0]

    dsigmaxx_dx = dsigma_xxddata[:,0,:,0]
    dsigmayy_dy = dsigma_yyddata[:,0,:,1]
    dsigmazz_dz = dsigma_zzddata[:,0,:,2]

    dsigmaxy_dx = dsigma_xyddata[:,0,:,0]
    dsigmaxy_dy = dsigma_xyddata[:,0,:,1]


    dsigmaxz_dx = dsigma_xzddata[:,0,:,0]

    dsigmaxz_dz = dsigma_xzddata[:,0,:,2]


    dsigmayz_dy = dsigma_yzddata[:,0,:,1]
    dsigmayz_dz = dsigma_yzddata[:,0,:,2]

    sigma_ij_i_x = dsigmaxx_dx + dsigmaxy_dy + dsigmaxz_dz
    sigma_ij_i_y = dsigmaxy_dx + dsigmayy_dy + dsigmayz_dz
    sigma_ij_i_z = dsigmaxz_dx + dsigmayz_dy + dsigmazz_dz

    equilibrium = torch.square(torch.mean(sigma_ij_i_x)) + torch.square(torch.mean(sigma_ij_i_y)) + torch.square(torch.mean(sigma_ij_i_z))

    epsilon = torch.stack([strain_source[:, :, 0], strain_source[:, :, 1], strain_source[:, :, 2], strain_source[:, :, 3], strain_source[:, :, 4], strain_source[:, :, 5]], dim=-1) #..// pred data
    sigma = torch.stack([sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz], dim=-1)
    strain_energy = torch.abs(torch.mean(strain_energy_density(epsilon, sigma)))

    return float(residual_strain), float(equilibrium), float(strain_energy)


def strain_energy_density(epsilon, sigma):
    if epsilon.shape[-1] != 6 or sigma.shape[-1] != 6:
        raise ValueError("应变张量和应力张量的最后一个维度必须为 6")

    epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xy, epsilon_xz, epsilon_yz = epsilon[..., 0], epsilon[..., 1], epsilon[..., 2], epsilon[..., 3], epsilon[..., 4], epsilon[..., 5]
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz = sigma[..., 0], sigma[..., 1], sigma[..., 2], sigma[..., 3], sigma[..., 4], sigma[..., 5]

    strain_energy = 0.5 * (
        epsilon_xx * sigma_xx +
        epsilon_yy * sigma_yy +
        epsilon_zz * sigma_zz +
        2 * epsilon_xy * sigma_xy +
        2 * epsilon_xz * sigma_xz +
        2 * epsilon_yz * sigma_yz
    )

    return strain_energy