import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
# loss0 = np.load('./results/PQNN_f20_PE_activation/loss.npy')
# loss1 = np.load('./results/PQNN_f20_QPE_activation/loss.npy')
# loss2 = np.load('./results/PQNN_f20_WPE_activation/loss.npy')
# loss3 = np.load('./results/PQNN_f20_WQPE_activation/loss.npy')
# # loss4 = np.load('./results/temp2/PQNN_f20_PE/loss.npy')
# loss5 = np.load('./results/temp2/PQNN_f20_QPE/loss.npy')
# loss6 = np.load('./results/temp2/PQNN_f20_WPE/loss.npy')
# loss7 = np.load('./results/temp2/DNN_f20_WQPE/loss.npy')

loss0 = np.load('./results/vmarm_f10/DNN_f10_PE_L_ns2/loss.npy')
loss1 = np.load('./results/vmarm_f10/DNN_f10_LAPE_L_ns2/loss.npy')
loss2 = np.load('./results/vmarm_f10/DNN_f10_LAPE_LPDI2_ns2/loss.npy')
loss3 = np.load('./results/vmarm_f10/QNN_f10_PE_L_ns2/loss.npy')
loss4 = np.load('./results/vmarm_f10/QNN_f10_LAPE_L_ns2/loss.npy')
loss5 = np.load('./results/vmarm_f10/QNN_f10_LAPE_LPDI_ns2/loss.npy')
# loss6 = np.load('./results/vsaltsm_f15/QNN_f15_LAPE_LG_vsaltsm/loss.npy')
# loss7 = np.load('./results/vsaltsm_f15/QNN_f15_LAPE_LGPDI_vsaltsm/loss.npy')

sio.savemat('./results_mat/vmarm_f10/loss_DNN_PE_L.mat', {'loss_pe': loss0})
sio.savemat('./results_mat/vmarm_f10/loss_DNN_LAPE_L.mat', {'loss_wpe': loss1})
sio.savemat('./results_mat/vmarm_f10/loss_DNN_LAPE_LPDI.mat', {'loss_wpe_pdi': loss2})
sio.savemat('./results_mat/vmarm_f10/loss_QNN_PE_L.mat', {'loss_qnn_pe': loss3})
sio.savemat('./results_mat/vmarm_f10/loss_QNN_LAPE_L.mat', {'loss_qnn_wpe': loss4})
sio.savemat('./results_mat/vmarm_f10/loss_QNN_LAPE_LPDI.mat', {'loss_qnn_wpe_pdi': loss5})
loss00=loss0[40:100000,][::100,]
loss11=loss1[40:100000,][::100,]
loss22=loss2[40:100000,][::100,]
loss33=loss3[40:100000,][::100,]
loss44=loss4[40:100000,][::100,]
loss55=loss5[40:100000,][::100,]
# loss66=loss6[40:40000,][::100,]
# loss77=loss7[40:40000,][::100,]
# loss11=loss0[10040:60000,][::200,]
# loss22=loss0[20040:70000,][::200,]



plt.plot(loss00,label='DNN_f10_PE_L',color='black')
plt.plot(loss11,label='DNN_f10_LAPE_L',color='magenta')
plt.plot(loss22,label='DNN_f10_LAPE_LPDI',color='orange')
plt.plot(loss33,label='QNN_f10_PE_L',color='red')
plt.plot(loss44,label='QNN_f10_LAPE_L',color='blue')
plt.plot(loss55,label='QNN_f10_LAPE_LPDI',color='green')
# plt.plot(loss66,label='QNN_f10_LAPE_LG',color='cyan')
# plt.plot(loss77,label='QNN_f10_LAPE_LGPDI',color='yellow')

# plt.xticks(
#     ticks=[0, 100-1, 200-1, 300-1, 400-1, 500-1, 600-1],  # 实际位置
#     labels=['0', '1', '2', '3', '4', '5', '6']           # 显示的标签
# )
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('x10000 iterations')
plt.legend()
plt.show()
