from ConfigParam import *

class PINN_PE():
    def __init__(self, inputs, PosEncoding, nettype, model, layers, loss_function, N_sample, lb, ub,wgs,device):

        # input settings
        self.inputs = inputs
        self.N_sample = N_sample
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        self.x = torch.tensor(inputs[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(inputs[:, 1:2], requires_grad=True).float().to(device)
        self.v = torch.tensor(inputs[:, 2:3], requires_grad=True).float().to(device)
        self.u0r = torch.tensor(inputs[:, 3:4], requires_grad=True).float().to(device)
        self.u0i = torch.tensor(inputs[:, 4:5], requires_grad=True).float().to(device)
        self.m = 1/(self.v ** 2)
        self.normx = 4 * (self.x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 2
        self.normy = 2 * (self.y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1
        self.sigma = 1 * (self.v-1.5) / 3 + 0.2
        self.dnn = model(layers, PosEncoding, nettype, self.normx[wgs,:].t(),self.normy[wgs,:].t(),self.sigma[wgs,:].t(),device).to(device)
        self.loss_function = loss_function


        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-20,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_u(self, x, y):
        u = self.dnn(4 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 2, 2*(y - self.lb[1]) / (self.ub[1] - self.lb[1])-1)
        return u

    def train(self, iter):
        self.dnn.train()
        print(device)
        Loss = []
        for epoch in range(iter):
            idx = np.random.choice(self.inputs.shape[0], self.N_sample, replace=False)
            idx_sum = idx
            x_batch = self.x[idx_sum, :]
            y_batch = self.y[idx_sum, :]
            m_batch = self.m[idx_sum, :]
            u0r_batch = self.u0r[idx_sum, :]
            u0i_batch = self.u0i[idx_sum, :]

            loss = self.loss_function(self.net_u, x_batch, y_batch, m_batch, u0r_batch, u0i_batch)
            self.optimizer_Adam.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()
            loss_tmp = loss.item()
            Loss.append(loss_tmp)
            if epoch < 1000:
                if epoch % 100 == 0:
                    print('Adam_It: %d, Loss: %.3e' % (epoch, loss_tmp,))
                    ur_pred = []
                    ui_pred = []
                    for i in range(200):
                        x_plt = X_plt[i * 200:(i + 1) * 200, :]
                        y_plt = Y_plt[i * 200:(i + 1) * 200, :]
                        ur_pr, ui_pr = self.predict(x_plt, y_plt)
                        ur_pred.append(ur_pr)
                        ui_pred.append(ui_pr)
                    ur_pred = np.vstack(ur_pred)
                    ui_pred = np.vstack(ui_pred)

                    ur_pred = np.reshape(ur_pred, (XX_plt.shape[0], XX_plt.shape[1]))
                    ui_pred = np.reshape(ui_pred, (XX_plt.shape[0], XX_plt.shape[1]))
                    fig = plt.figure()
                    plt.imshow(ur_pred, cmap='bwr')
                    plt.colorbar()
                    plt.savefig(result_dir + '/ur_it_' + str(epoch) + '.png')
                    plt.close(fig)
                    fig = plt.figure()
                    plt.imshow(ui_pred, cmap='bwr')
                    plt.colorbar()
                    plt.savefig(result_dir + '/ui_it_' + str(epoch) + '.png')
                    plt.close(fig)
                if epoch == 0:
                    torch.save(self.dnn.state_dict(), result_dir + '/nnparam' + str(epoch) + '.pt')
            elif epoch < 10000:
                if epoch % 1000 == 0:
                    print('Adam_It: %d, Loss: %.3e' % (epoch, loss_tmp,))
                    ur_pred = []
                    ui_pred = []
                    for i in range(200):
                        x_plt = X_plt[i * 200:(i + 1) * 200, :]
                        y_plt = Y_plt[i * 200:(i + 1) * 200, :]
                        ur_pr, ui_pr = self.predict(x_plt, y_plt)
                        ur_pred.append(ur_pr)
                        ui_pred.append(ui_pr)
                    ur_pred = np.vstack(ur_pred)
                    ui_pred = np.vstack(ui_pred)
                    ur_pred = np.reshape(ur_pred, (XX_plt.shape[0], XX_plt.shape[1]))
                    ui_pred = np.reshape(ui_pred, (XX_plt.shape[0], XX_plt.shape[1]))
                    fig = plt.figure()
                    plt.imshow(ur_pred, cmap='bwr')
                    plt.colorbar()
                    plt.savefig(result_dir + '/ur_it_' + str(epoch) + '.png')
                    plt.close(fig)
                    fig = plt.figure()
                    plt.imshow(ui_pred, cmap='bwr')
                    plt.colorbar()
                    plt.savefig(result_dir + '/ui_it_' + str(epoch) + '.png')
                    plt.close(fig)
            elif epoch <= 100000:
                if epoch % 10000 == 0:
                    print('Adam_It: %d, Loss: %.3e' % (epoch, loss_tmp,))
                    ur_pred = []
                    ui_pred = []
                    for i in range(200):
                        x_plt = X_plt[i * 200:(i + 1) * 200, :]
                        y_plt = Y_plt[i * 200:(i + 1) * 200, :]
                        ur_pr, ui_pr = self.predict(x_plt, y_plt)
                        ur_pred.append(ur_pr)
                        ui_pred.append(ui_pr)
                    ur_pred = np.vstack(ur_pred)
                    ui_pred = np.vstack(ui_pred)
                    ur_pred = np.reshape(ur_pred, (XX_plt.shape[0], XX_plt.shape[1]))
                    ui_pred = np.reshape(ui_pred, (XX_plt.shape[0], XX_plt.shape[1]))
                    fig = plt.figure()
                    plt.imshow(ur_pred, cmap='bwr')
                    plt.colorbar()
                    plt.savefig(result_dir + '/ur_it_' + str(epoch) + '.png')
                    plt.close(fig)
                    fig = plt.figure()
                    plt.imshow(ui_pred, cmap='bwr')
                    plt.colorbar()
                    plt.savefig(result_dir + '/ui_it_' + str(epoch) + '.png')
                    plt.close(fig)
                    torch.save(self.dnn.state_dict(), result_dir + '/nnparam' + str(epoch) + '.pt')
                    lost_np = np.array(Loss)
                    fig = plt.figure()
                    plt.plot(lost_np)
                    plt.yscale('log')
                    plt.ylabel('loss')
                    plt.xlabel('iterations')
                    plt.savefig(result_dir + '/loss.png')
                    plt.close(fig)
                    np.save(result_dir + '/loss.npy', lost_np)
        return Loss

    def predict(self, xx, yy):

        self.dnn.eval()
        u = self.net_u(xx, yy)
        ur = u[:, 0:1].detach().cpu().numpy()
        ui = u[:, 1:2].detach().cpu().numpy()
        return ur, ui

    def train_LBFGS(self):
        loss = self.loss_function(self.net_u, self.x, self.y, self.m, self.u0r, self.u0i)
        # self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        Loss = []
        self.iter += 1
        loss_tmp = loss.item()
        Loss.append(loss_tmp)
        if self.iter % 1000 == 0:
            print('LBFGS_It: %d, Loss: %.3e' %(self.iter, loss.item(),))
        return loss

