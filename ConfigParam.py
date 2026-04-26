from ConfigLib import *

sys.path.insert(0, './core')
data_dir = './data'
result_dir = './results/QNN_f15_PE_L_vsaltsm'
if not os.path.exists(result_dir+'/'):
    os.makedirs(result_dir+'/')

# print(f"PyTorch版本: {torch.__version__}")
# print(f"CUDA是否可用: {torch.cuda.is_available()}")
# print(f"检测到的GPU数量: {torch.cuda.device_count()}")
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_per_process_memory_fraction(0.8, device=device)


# layers = [2, 80, 80, 80, 80, 80, 2]
layers = [2, 45, 45, 45, 45, 45, 2]
# layers = [2, 50, 50, 50, 50, 50, 2]

# layers = [2, 100, 100, 100, 100, 100, 2]
# layers = [2, 60, 60, 60, 60, 60, 2]

# layers = [2, 140, 140, 140, 140, 140, 2]
# layers = [2, 80, 80, 80, 80, 80, 2]

N_sample = 2000
f = 15
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
#######################################################
nettype = "QNN"                                 # DNN / /QNN
equation = "sc_pml_attenuation"                  # sc / sc_pml / sc_attenuation / sc_pml_attenuation
pml = "y"                                        # y / n
attenuation = "y"                                # y / n
PosEncoding = 'pe_l'                           # lape_l / lape_lq / lape_lg / pe_q / pe_l / n
#######################################################
# deep neural networks and losses

from core import sc_pml_attenuation as EL
loss_func = EL.equation_loss

from core import MLPs_with_PE
model = MLPs_with_PE.Model

#######################################################

#######################################################
# v list:
# vm00pml.mat: linear model
# vm11pml.mat: smoothed marmousi
# vm22pml.mat: slightly smoothed marmousi
# vmpml.mat: marmousi
#
# u0 list:
# um0pml.mat: background wave field
# um0pmlq.mat: attenuated background wave field
# u0free.mat: background wave field with freesurface boundary condition
# u0free_noa.mat: attenuated background wave field with freesurface boundary condition

data = scipy.io.loadmat(data_dir+'/vsaltsm.mat')
vtrpml = data['vsaltsm']
# data = scipy.io.loadmat(data_dir+'/vsm_wpad.mat')
# v_wpad = data['vsm_wpad']

def weighted_systematic_sampling_2d_randomized(weight_flat, M):
    """
    随机化系统采样（分层随机采样）
    参数:
        weight_flat: 一维数组，长度 N，归一化权重（概率）
        M: 需要的采样点数
    返回:
        indices: 长度为 M 的一维索引（在展平数组中的位置）
    """
    N = len(weight_flat)
    cumsum = np.cumsum(weight_flat)
    # plt.plot(cumsum)
    # plt.show()
    indices = np.empty(M, dtype=int)
    # 将 [0,1) 等分为 M 个区间
    for i in range(M):
        # 第 i 层的区间为 [i/M, (i+1)/M)
        low = i / M
        high = (i + 1) / M
        # 随机阈值
        t = np.random.uniform(low, high)
        idx = np.searchsorted(cumsum, t)
        # 边界保护（由于浮点误差，idx 可能等于 N，此时取 N-1）
        idx = min(idx, N - 1)
        indices[i] = idx

    # 理论上不会重复，但若因权重为零导致重复，做微小处理
    unique, counts = np.unique(indices, return_counts=True)
    if len(unique) == M:
        return indices
    else:
        # 极少数情况（如某层内总概率为0），从未选中点中按权重补采
        chosen = set(unique)
        remaining_probs = weight_flat.copy()
        remaining_probs[list(chosen)] = 0
        remaining_probs /= remaining_probs.sum()
        missing = M - len(unique)
        additional = np.random.choice(N, size=missing, replace=False, p=remaining_probs)
        return np.concatenate([unique, additional])
epsilon = 1e-6
weight = 1.0 / (vtrpml**2 + epsilon)
weight /= weight.sum()
weight_flat = weight.ravel()
M = 100
wgs = weighted_systematic_sampling_2d_randomized(weight_flat, M)

x = np.arange(1,Nx+1)
y = np.arange(1,Ny+1)
XX, YY = np.meshgrid(x, y)
x_gs = XX.flatten()[:, ][wgs]
y_gs = YY.flatten()[:, ][wgs]
scipy.io.savemat(result_dir + '/x_rand.mat', {'x_ra': x_gs})
scipy.io.savemat(result_dir + '/y_rand.mat', {'y_ra': y_gs})

vtr = vtrpml[N_pml:Ny-N_pml,N_pml:Nx-N_pml]
x = x_step*np.arange(Nx)
y = y_step*np.arange(Ny)
XX, YY = np.meshgrid(x, y)
XX_plt = np.float64(XX[N_pml:Ny - N_pml:10, N_pml:Nx - N_pml:10])
YY_plt = np.float64(YY[N_pml:Ny - N_pml:10, N_pml:Nx - N_pml:10])
X_plt = torch.tensor(XX_plt.flatten()[:, None], requires_grad=True).float().to(device)
Y_plt = torch.tensor(YY_plt.flatten()[:, None], requires_grad=True).float().to(device)

if pml == "n":
    if attenuation == "n":
        data = scipy.io.loadmat(data_dir + '/um0pml.mat')
        u0pml = data['um0']
    elif attenuation == "y":
        data = scipy.io.loadmat(data_dir + '/um0pmlq.mat')
        u0pml = data['um0pmlq']
    u0 = u0pml[N_pml:Ny - N_pml, N_pml:Nx - N_pml]
    u0r = np.real(u0)
    u0i = np.imag(u0)
    u0_in = np.hstack((u0r.flatten()[:, None], u0i.flatten()[:, None]))
    vu0_in = np.hstack((vtr.flatten()[:, None], u0_in))
    XX_nopml = XX[N_pml:Ny-N_pml,N_pml:Nx-N_pml]
    YY_nopml = YY[N_pml:Ny-N_pml,N_pml:Nx-N_pml]
    X_in = np.hstack((XX_nopml.flatten()[:, None], YY_nopml.flatten()[:, None]))
    inputs_full = np.hstack((X_in, vu0_in))
elif pml == 'y':
    if attenuation == "n":
        data = scipy.io.loadmat(data_dir + '/um0pml.mat')
        u0pml = data['um0']
    else:
        data = scipy.io.loadmat(data_dir + '/u0_q50_f15_2000x2000.mat')
        u0pml = data['u0_q50_f15_2000x2000']
    u0r = np.real(u0pml)
    u0i = np.imag(u0pml)
    u0_in = np.hstack((u0r.flatten()[:, None], u0i.flatten()[:, None]))
    vu0_in = np.hstack((vtrpml.flatten()[:, None], u0_in))
    X_in = np.hstack((XX.flatten()[:, None], YY.flatten()[:, None]))
    inputs_full = np.hstack((X_in, vu0_in))


# XX_plt = np.float64(XX)
# YY_plt = np.float64(YY)
# X_plt = np.hstack((XX_plt.flatten()[:, None], YY_plt.flatten()[:, None]))
lb = X_in.min(0)
ub = X_in.max(0)

