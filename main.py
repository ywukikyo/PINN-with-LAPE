from ConfigParam import *

from core.PINN import PINN_PE

pinn = PINN_PE(inputs_full, PosEncoding, nettype, model, layers, loss_func, N_sample, lb, ub,wgs,device)

# pinn.dnn.load_state_dict(torch.load(data_dir  + '/L_nnparam40000.pt',map_location=device))

Loss = pinn.train(nIter)

lost_np = np.array(Loss)
fig = plt.figure()
plt.plot(lost_np)
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.savefig(result_dir + '/loss.png')
plt.close(fig)
np.save(result_dir + '/loss.npy', lost_np)

