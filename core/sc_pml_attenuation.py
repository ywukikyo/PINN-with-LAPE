import torch.nn.functional as F
from ConfigParam import *

def equation_loss(net_u, x, y, m, u0r, u0i):
    """ The pytorch autograd version of calculating residual """
    a0 = 1.79
    f0 = 10
    C = a0 * f0 / f
    m0 = 1/(v0 ** 2)
    alpha = 1 / Q
    rhot = (1 - alpha / np.pi * np.log(f / fr) - 1j * alpha / 2) ** 2
    m0r = m0 * np.real(rhot)
    m0i = m0 * np.imag(rhot)
    mr = m * np.real(rhot)
    mi = m * np.imag(rhot)
    u = net_u(x, y)
    ur = u[:, 0:1]
    ui = u[:, 1:2]

    # pml setting
    lx = F.relu((N_pml * x_step - x - 0.5 * x_step) / (N_pml * x_step)) + F.relu((x - (Nx - N_pml) * x_step + 0.5 * x_step) / (N_pml * x_step))
    ly = F.relu((N_pml * y_step - y - 0.5 * y_step) / (N_pml * y_step)) + F.relu((y - (Nx - N_pml) * y_step + 0.5 * y_step) / (N_pml * y_step))
    # lx = F.relu((N_pml * x_step - x - x_step) / (N_pml * x_step)) + F.relu((x - (Nx - N_pml) * x_step) / (N_pml * x_step))
    # ly = F.relu((N_pml * y_step - y - y_step) / (N_pml * y_step)) + F.relu((y - (Nx - N_pml) * y_step) / (N_pml * y_step))
    pml_tmp1 = C ** 2 * lx ** 2 * ly ** 2
    pml_tmp2 = C ** 2 * lx ** 4
    pml_tmp3 = C ** 2 * ly ** 4
    pml_tmp4 = C * (ly ** 2 - lx ** 2)
    pml_tmp5 = C * (lx ** 2 + ly ** 2)

    ur_x = torch.autograd.grad(
        ur, x,
        grad_outputs=torch.ones_like(ur),
        retain_graph=True,
        create_graph=True
    )[0]
    ur_y = torch.autograd.grad(
        ur, y,
        grad_outputs=torch.ones_like(ur),
        retain_graph=True,
        create_graph=True
    )[0]
    ui_x = torch.autograd.grad(
        ui, x,
        grad_outputs=torch.ones_like(ui),
        retain_graph=True,
        create_graph=True
    )[0]
    ui_y = torch.autograd.grad(
        ui, y,
        grad_outputs=torch.ones_like(ui),
        retain_graph=True,
        create_graph=True
    )[0]

    eu_xr = (1+pml_tmp1)/(1+pml_tmp2)*ur_x+pml_tmp4/(1+pml_tmp2)*ui_x
    eu_yr = (1+pml_tmp1)/(1+pml_tmp3)*ur_y-pml_tmp4/(1+pml_tmp3)*ui_y
    eu_xi = -pml_tmp4/(1+pml_tmp2)*ur_x+(1+pml_tmp1)/(1+pml_tmp2)*ui_x
    eu_yi = pml_tmp4/(1+pml_tmp3)*ur_y+(1+pml_tmp1)/(1+pml_tmp3)*ui_y
    ur_xx = torch.autograd.grad(
        eu_xr, x,
        grad_outputs=torch.ones_like(eu_xr),
        retain_graph=True,
        create_graph=True
    )[0]
    ur_yy = torch.autograd.grad(
        eu_yr, y,
        grad_outputs=torch.ones_like(eu_yr),
        retain_graph=True,
        create_graph=True
    )[0]
    ui_xx = torch.autograd.grad(
        eu_xi, x,
        grad_outputs=torch.ones_like(eu_xi),
        retain_graph=True,
        create_graph=True
    )[0]
    ui_yy = torch.autograd.grad(
        eu_yi, y,
        grad_outputs=torch.ones_like(eu_yi),
        retain_graph=True,
        create_graph=True
    )[0]

    ur_r = (1 - pml_tmp1) * (2 * math.pi * f * 1e-3) ** 2 * (mr * (ur + u0r) - mi * (ui + u0i))
    ui_r = pml_tmp5 * (2 * math.pi * f * 1e-3) ** 2 * (mr * (ui + u0i) + mi * (ur + u0r))
    u0r_r = (1 - pml_tmp1) * (2 * math.pi * f * 1e-3) ** 2 * (-m0r * u0r + m0i * u0i)
    u0i_r = pml_tmp5 * (2 * math.pi * f * 1e-3) ** 2 * (-m0r * u0i - m0i * u0r)

    ur_i = (-pml_tmp5) * (2 * math.pi * f * 1e-3) ** 2 * ( mr * (ur + u0r) - mi * (ui + u0i))
    ui_i = (1 - pml_tmp1) * (2 * math.pi * f * 1e-3) ** 2 * (mr * (ui + u0i) + mi * (ur + u0r))
    u0r_i = (-pml_tmp5) * (2 * math.pi * f * 1e-3) ** 2 * (-m0r * u0r + m0i * u0i)
    u0i_i = (1 - pml_tmp1) * (2 * math.pi * f * 1e-3) ** 2 * (-m0r * u0i - m0i * u0r)

    f_r = ur_xx + ur_yy + ur_r + ui_r + u0r_r + u0i_r
    f_i = ui_xx + ui_yy + ur_i + ui_i + u0r_i + u0i_i

    loss_f = torch.sum(f_r ** 2) + torch.sum(f_i ** 2)
    return loss_f