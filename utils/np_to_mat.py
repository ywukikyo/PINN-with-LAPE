import scipy.io as sio
from ConfigLib import *
import torch.nn.functional as F

sys.path.insert(0, './core')
data_dir = './data'
velocity = 'vmarm_f10'
test = '/QNN_f10_LAPE_LPDI2_ns2'
results_dir = './results/' + velocity + test
results_mat_dir = './results_mat/' + velocity + test
if not os.path.exists(results_mat_dir+'/'):
    os.makedirs(results_mat_dir+'/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# layers = [2, 80, 80, 80, 80, 80, 2]
# layers = [2, 45, 45, 45, 45, 45, 2]
layers = [2, 50, 50, 50,  50, 2]
# layers = [2, 80, 80, 80, 80, 80, 2]

# layers = [2, 100, 100, 100, 100, 100, 2]
# layers = [2, 60, 60, 60, 60, 60, 2]

# layers = [2, 140, 140, 140, 140, 140, 2]
# layers = [2, 80, 80, 80, 80,  2]
N_sample = 2000
f = 10
Nx = 2400
Ny = 2400
N_pml = 200
x_step = 2
y_step = 1
v0 = 1.5
Q = 50
fr = 50
nIter = 100001
sigmax = 2 * 5
sigmay = 2 * 5
a0 = 1.79
f0 = 10
C = a0 * f0 / f
#######################################################
nettype = "QNN"                                 # DNN / /QNN
equation = "sc_pml_attenuation"                  # sc / sc_pml / sc_attenuation / sc_pml_attenuation
pml = "y"                                        # y / n
attenuation = "y"                                # y / n
PosEncoding = 'lape_l'                           # lape_l / lape_lq / lape_lg / pe_q / pe_l / n

data = scipy.io.loadmat(data_dir + '/u0_q50_f10_2000x2000.mat')
u0pml = data['u0_q50_f10_2000x2000']
#######################################################
# deep neural networks and losses

from core import MLPs_with_PE
model = MLPs_with_PE.Model

data = scipy.io.loadmat(data_dir+'/vgeometry.mat')
vtrpml = data['vgeometry']

index = np.arange(Nx*Ny)
# wgs = np.random.choice(index, 80, p=v_wpad,replace=False)
wgs = np.random.choice(index, 100, replace=False)

vtr = vtrpml[N_pml:Ny-N_pml,N_pml:Nx-N_pml]
x = x_step*np.arange(Nx)
y = y_step*np.arange(Ny)
XX, YY = np.meshgrid(x, y)


u0r = np.real(u0pml)
u0i = np.imag(u0pml)
u0_in = np.hstack((u0r.flatten()[:, None], u0i.flatten()[:, None]))
vu0_in = np.hstack((vtrpml.flatten()[:, None], u0_in))
X_in = np.hstack((XX.flatten()[:, None], YY.flatten()[:, None]))
inputs_full = np.hstack((X_in, vu0_in))
lb = X_in.min(0)
ub = X_in.max(0)

xtensor = torch.tensor(inputs_full[:, 0:1], requires_grad=True).float().to(device)
ytensor = torch.tensor(inputs_full[:, 1:2], requires_grad=True).float().to(device)
vtensor = torch.tensor(inputs_full[:, 2:3], requires_grad=True).float().to(device)
lb = torch.tensor(lb).float().to(device)
ub = torch.tensor(ub).float().to(device)
if y_step == 1:
    normx = 4 * (xtensor - lb[0]) / (ub[0] - lb[0]) - 2
    normy = 2 * (ytensor - lb[1]) / (ub[1] - lb[1]) - 1
elif y_step == 2:
    normx = 2 * (xtensor - lb[0]) / (ub[0] - lb[0]) - 1
    normy = 2 * (ytensor - lb[1]) / (ub[1] - lb[1]) - 1
sigma = 1.8*(vtensor-1.5)/3+0.2
pinn = model(layers,PosEncoding, nettype, normx[wgs,:].t(),normy[wgs,:].t(),sigma[wgs,:].t(),device).to(device)
pinn.load_state_dict(torch.load(results_dir + '/nnparam50000.pt',map_location=torch.device('cpu'))) #= torch.load(results_dir + '/nnparam.pt',map_location=torch.device('cpu'))
pinn.eval()

# x_full = torch.tensor(inputs_full[:, 0:1], requires_grad=True).float().to(device)
# y_full = torch.tensor(inputs_full[:, 1:2], requires_grad=True).float().to(device)
# v_full = torch.tensor(inputs_full[:, 2:3], requires_grad=True).float().to(device)
# X = torch.tensor(np.float64(XX.flatten()[:, None]), requires_grad=True).float().to(device)
# Y = torch.tensor(np.float64(YY.flatten()[:, None]), requires_grad=True).float().to(device)
XX_plt = np.float64(XX[0:-1:10, 0:-1:10])
YY_plt = np.float64(YY[0:-1:10, 0:-1:10])
X_plt = torch.tensor(XX_plt.flatten()[:, None], requires_grad=True).float().to(device)
Y_plt = torch.tensor(YY_plt.flatten()[:, None], requires_grad=True).float().to(device)


ur_pinn = []
ui_pinn = []
ur_la_pinn = []
ui_la_pinn = []
for i in range(240):
    x = X_plt[i*240:(i+1)*240,:]
    y = Y_plt[i*240:(i+1)*240,:]
    if y_step == 1:
        x_norm = (4 * (x - lb[0]) / (ub[0] - lb[0]) - 2).to(device)
        y_norm = (2 * (y - lb[1]) / (ub[1] - lb[1]) - 1).to(device)
    elif y_step == 2:
        x_norm = (2 * (x - lb[0]) / (ub[0] - lb[0]) - 1).to(device)
        y_norm = (2 * (y - lb[1]) / (ub[1] - lb[1]) - 1).to(device)
    u = pinn(x_norm, y_norm)
    ur = u[:, 0:1]
    ui = u[:, 1:2]
    # pml setting
    # lx = F.relu((9.5 * 40 - x) / (9.5 * 40)) + F.relu((x - 109.5 * 40) / (9.5 * 40))
    # ly = F.relu((9.5 * 20 - y) / (9.5 * 20)) + F.relu((y - 109.5 * 20) / (9.5 * 20))
    lx = F.relu((N_pml * x_step - x - 0.5 * x_step) / (N_pml * x_step)) + F.relu((x - (Nx - N_pml) * x_step + 0.5 * x_step) / (N_pml * x_step))
    ly = F.relu((N_pml * y_step - y - 0.5 * y_step) / (N_pml * y_step)) + F.relu((y - (Nx - N_pml) * y_step + 0.5 * y_step) / (N_pml * y_step))
    pml_tmp1 = C ** 2 * lx ** 2 * ly ** 2
    pml_tmp2 = C ** 2 * lx ** 4
    pml_tmp3 = C ** 2 * ly ** 4
    pml_tmp4 = C * (ly ** 2 - lx ** 2)
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
    eu_xr = (1 + pml_tmp1) / (1 + pml_tmp2) * ur_x + pml_tmp4 / (1 + pml_tmp2) * ui_x
    eu_yr = (1 + pml_tmp1) / (1 + pml_tmp3) * ur_y - pml_tmp4 / (1 + pml_tmp3) * ui_y
    eu_xi = -pml_tmp4 / (1 + pml_tmp2) * ur_x + (1 + pml_tmp1) / (1 + pml_tmp2) * ui_x
    eu_yi = pml_tmp4 / (1 + pml_tmp3) * ur_y + (1 + pml_tmp1) / (1 + pml_tmp3) * ui_y
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
    ur_la = ur_xx + ur_yy
    ui_la = ui_xx + ui_yy
    ur_pinn_batch = u[:, 0:1].detach().cpu().numpy()
    ui_pinn_batch = u[:, 1:2].detach().cpu().numpy()
    ur_la_pinn_batch = ur_la.detach().cpu().numpy()
    ui_la_pinn_batch = ui_la.detach().cpu().numpy()
    ur_pinn.append(ur_pinn_batch)
    ui_pinn.append(ui_pinn_batch)
    ur_la_pinn.append(ur_la_pinn_batch)
    ui_la_pinn.append(ui_la_pinn_batch)
ur_pinn = np.vstack(ur_pinn)
ui_pinn = np.vstack(ui_pinn)
ur_la_pinn = np.vstack(ur_la_pinn)
ui_la_pinn = np.vstack(ui_la_pinn)

alpha = 1/Q
# alpha = 0
alpha0 = 1 / Q
rhot = (1 - alpha / np.pi * np.log(f / fr) - 1j * alpha / 2) ** 2
rhot0 = (1 - alpha0 / np.pi * np.log(f / fr) - 1j * alpha0 / 2) ** 2
m0 = 1 / v0 ** 2 * rhot0

Lpml = N_pml/10
f0 = 10
omega = 1e-3 * 2 * math.pi * f
beta = 2 * math.pi * 1.79 * f0 / f
xc = np.arange(1, Nx/10 + 1)
yc = np.arange(1, Ny/10 + 1)
pmly = lambda x: 1 - 1j * beta * ((Lpml - x + 1 / 2) / Lpml) ** 2 * (Lpml - x + 1 / 2 > 0) - 1j * beta * (
        (x - 1 / 2 - Ny + Lpml) / Lpml) ** 2 * (x - 1 / 2 - Ny + Lpml > 0)
pmlx = lambda x: 1 - 1j * beta * ((Lpml - x + 1 / 2) / Lpml) ** 2 * (Lpml - x + 1 / 2 > 0) - 1j * beta * (
        (x - 1 / 2 - Nx + Lpml) / Lpml) ** 2 * (x - 1 / 2 - Nx + Lpml > 0)
gridX, gridY = np.meshgrid(xc, yc)
C0 = np.multiply(pmly(gridY), pmlx(gridX))
exey = np.reshape(C0, (int(Nx * Ny/100), 1))

u0 = u0pml[0:-1:10,0:-1:10].flatten()[:,None]
m = -((ur_la_pinn + 1j * ui_la_pinn) - omega ** 2 * m0 * exey * u0) / (omega ** 2 * exey * (ur_pinn + 1j *ui_pinn + u0))

vfu = np.reshape(np.real(np.sqrt(rhot / m)), (int(Nx/10), int(Ny/10)))
ur_pinn = np.reshape(ur_pinn, (XX_plt.shape[0], XX_plt.shape[1]))
ui_pinn = np.reshape(ui_pinn, (XX_plt.shape[0], XX_plt.shape[1]))
u_pinn = ur_pinn + 1j * ui_pinn
sio.savemat(results_mat_dir + '/u.mat', {'u_pinn': u_pinn})
sio.savemat(results_mat_dir + '/vfu.mat', {'vfu': vfu})







