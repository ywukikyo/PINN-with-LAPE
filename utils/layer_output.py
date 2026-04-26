import scipy.io as sio
from ConfigLib import *

sys.path.insert(0, './core')
data_dir = './data'
velocity = 'vsalt'
test = '/QNN_LAPE_LGPDI_vsaltsm'
results_dir = './results/' + velocity + test
results_mat_dir = './results_mat/' + velocity + test
if not os.path.exists(results_mat_dir+'/'):
    os.makedirs(results_mat_dir+'/')
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# layers = [2, 80, 80, 80, 80, 80, 2]
layers = [2, 45, 45, 45, 45, 45, 2]
# layers = [2, 140, 140, 140, 140, 140, 2]
# layers = [2, 80, 80, 80, 80, 2]
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
PosEncoding = 'lape_lgpdi'                           # lape_l / lape_lq / lape_lg / pe_q / pe_l / n

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
wgs = np.random.choice(index, 80, replace=False)

vtr = vtrpml[N_pml:Ny-N_pml,N_pml:Nx-N_pml]
x = x_step*np.arange(Nx)
y = y_step*np.arange(Ny)
XX, YY = np.meshgrid(x, y)

# ur_pinn = np.load(results_dir + '/ur_pinn.npy')
# ui_pinn = np.load(results_dir + '/ui_pinn.npy')
# u_pinn = ur_pinn + 1j * ui_pinn
# sio.savemat(results_mat_dir + '_u.mat', {'u_pinn': u_pinn})

XX_plt = np.float64(XX[N_pml:Ny - N_pml:10, N_pml:Nx - N_pml:10])
YY_plt = np.float64(YY[N_pml:Ny - N_pml:10, N_pml:Nx - N_pml:10])
X_plt = torch.tensor(XX_plt.flatten()[:, None], requires_grad=True).float().to(device)
Y_plt = torch.tensor(YY_plt.flatten()[:, None], requires_grad=True).float().to(device)
u0r = np.real(u0pml)
u0i = np.imag(u0pml)
u0_in = np.hstack((u0r.flatten()[:, None], u0i.flatten()[:, None]))
vu0_in = np.hstack((vtrpml.flatten()[:, None], u0_in))
X_in = np.hstack((XX.flatten()[:, None], YY.flatten()[:, None]))
inputs_full = np.hstack((X_in, vu0_in))
lb = X_in.min(0)
ub = X_in.max(0)

inputs = inputs_full
N_sample = N_sample
lb = torch.tensor(lb).float().to(device)
ub = torch.tensor(ub).float().to(device)
x = torch.tensor(inputs[:, 0:1], requires_grad=True).float().to(device)
y = torch.tensor(inputs[:, 1:2], requires_grad=True).float().to(device)
v = torch.tensor(inputs[:, 2:3], requires_grad=True).float().to(device)
u0r = torch.tensor(inputs[:, 3:4], requires_grad=True).float().to(device)
u0i = torch.tensor(inputs[:, 4:5], requires_grad=True).float().to(device)
m = 1/(v ** 2)
normx = 4 * (x - lb[0]) / (ub[0] - lb[0]) - 2
normy = 2 * (y - lb[1]) / (ub[1] - lb[1]) - 1
sigma = 1.8*(v-1.5)/3+0.2

pinn = model(layers, PosEncoding, nettype, normx[wgs,:].t(), normy[wgs,:].t(), sigma[wgs,:].t()).to(device)
pinn.load_state_dict(torch.load(results_dir + '/nnparam0.pt',map_location=torch.device('cpu'))) #= torch.load(results_dir + '/nnparam.pt',map_location=torch.device('cpu'))
pinn.eval()

XX_plt = np.float64(XX[0:-1:10, 0:-1:10])
YY_plt = np.float64(YY[0:-1:10, 0:-1:10])
X_plt = torch.tensor(XX_plt.flatten()[:, None], requires_grad=True).float().to(device)
Y_plt = torch.tensor(YY_plt.flatten()[:, None], requires_grad=True).float().to(device)


pinnxc = pinn.lay_pe.xc.detach().cpu().numpy()
pinnyc = pinn.lay_pe.yc.detach().cpu().numpy()
pinnsigma_x = pinn.lay_pe.sigma_x.detach().cpu().numpy()
pinnsigma_y = pinn.lay_pe.sigma_y.detach().cpu().numpy()

sio.savemat(results_mat_dir + '/pinnxc.mat', {'pinnxc': pinnxc})
sio.savemat(results_mat_dir + '/pinnyc.mat', {'pinnyc': pinnyc})
sio.savemat(results_mat_dir + '/pinnsigma_x.mat', {'pinnsigma_x': pinnsigma_x})
sio.savemat(results_mat_dir + '/pinnsigma_y.mat', {'pinnsigma_y': pinnsigma_y})


# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()  # 分离计算图，避免内存泄漏
#     return hook
# model.lay_pe.register_forward_hook(get_activation('lay_pe'))


# for j in range(80):
#     neuron_output = []
#     for i in range(240):
#         x = X_plt[i * 240:(i + 1) * 240, :]
#         y = Y_plt[i * 240:(i + 1) * 240, :]
#         x_norm = (4 * (x - lb[0]) / (ub[0] - lb[0]) - 2).to(device)
#         y_norm = (2 * (y - lb[1]) / (ub[1] - lb[1]) - 1).to(device)
#         u = pinn(x_norm, y_norm)
#         layer_output = pinn.gaus
#         # layer_output = activation['lay_pe']
#         neuron_batch = layer_output[:, j].detach().cpu().numpy()
#         neuron_output.append(neuron_batch)
#     neuron_output = np.vstack(neuron_output)
#     neuron_output = np.reshape(neuron_output, (XX_plt.shape[0], XX_plt.shape[1]))
#     sio.savemat(results_mat_dir + '/WPE_neuron' + str(j) + '.mat', {'WPE_neuron': neuron_output})
