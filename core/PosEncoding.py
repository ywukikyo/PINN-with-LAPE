import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import Parameter, init

use_gpu = torch.cuda.is_available()

class PosEncoding_f(nn.Module):
    def __init__(self, in_features: int, num_freq: int, bias: bool = True):
        super(PosEncoding_f, self).__init__()
        self.f = 10
        self.in_features = in_features
        self.num_freq = num_freq
        self.out_features = 2 * in_features * num_freq + in_features

    def forward(self, x, y):
        wave_num = 6 * torch.pi
        # wave_num = 14 * torch.pi
        # sinx = torch.sin(x * (2 ** (- 5) * wave_num))
        # cosx = torch.cos(x * (2 ** (- 5) * wave_num))
        # x_pe = torch.cat((x, y, sinx, cosx), axis=-1)
        x_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            x_pe = torch.cat((x_pe, sinx, cosx, siny, cosy), axis=-1)
        return x_pe

class PosEncoding_QPE(nn.Module):
    def __init__(self, in_features: int, out_features: int,num_freq: int,bias: bool = True, device = None):
        super(PosEncoding_QPE, self).__init__()
        self.f = 10
        self.num_freq = num_freq
        self.in_features = num_freq*2+1
        self.out_features = out_features
        self.weight_pex = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pey = Parameter(torch.Tensor(self.in_features, out_features))

        if bias:
            self.bias_pex = Parameter(torch.Tensor(out_features))
            self.bias_pey = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pex.to(device)
            self.weight_pey.to(device)
            self.bias_pex.to(device)
            self.bias_pey.to(device)
            self.bias_b.to(device)
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pex, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pey, a=math.sqrt(5))
        if self.bias_pex is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pex)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pex, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pey)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pey, -bound, bound)

    def forward(self, x, y):
        # wave_num_gaus = 2 * torch.pi * self.f / v_gaus
        wave_num = 6 * torch.pi
        x_pe = x
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq+1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq+1) * wave_num))
            x_pe = torch.cat((x_pe, sinx, cosx), axis=-1)
        y_pe = y
        for i in range(self.num_freq):
            siny = torch.sin(y * (2 ** (i - self.num_freq+1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq+1) * wave_num))
            y_pe = torch.cat((y_pe, siny, cosy), axis=-1)
        out = (torch.matmul(x_pe, self.weight_pex) + self.bias_pex) * (torch.matmul(y_pe, self.weight_pey) + self.bias_pey) + self.bias_b
        return out

class PosEncoding_PE_sigmoid(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_PE_sigmoid, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.bias_pe.to(device)
            self.bias_b.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 14*torch.pi

        xy = torch.cat((x, y), axis=-1)
        xy_pe = xy
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        xy_sig = torch.sigmoid(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
                        x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))
        out = xy_pe * (torch.matmul(xy_sig, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_L(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WPE_L, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 50
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.1, 1)
        init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y):
        wave_num = 6*torch.pi
        # wave_num = 14 * torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_LI(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device=None):
        super(PosEncoding_WPE_LI, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.1, 1)
        init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        start_x = -2 + 1 / 3  # 起始值 = -4/3
        ends_x = 2 - 1 / 3
        step_x = 5 / 6  # 间隔 1/3

        start_y = -1 + 1 / 6  # 起始值 = -4/3
        ends_y = 1 - 1 / 6
        step_y = 5 / 12  # 间隔 1/3
        n = 5

        with torch.no_grad():
            values_x = torch.arange(start_x, start_x + n * step_x, step_x)
            values_y = torch.arange(start_y, start_y + n * step_y, step_y)
            self.xc.data[0, 0:n] = start_x
            self.xc.data[0, n:2 * n] = values_x
            self.xc.data[0, 2 * n:3 * n] = ends_x
            self.xc.data[0, 3 * n:4 * n] = values_x

            self.yc.data[0, 0:n] = values_y
            self.yc.data[0, n:2 * n] = start_y
            self.yc.data[0, 2 * n:3 * n] = values_y
            self.yc.data[0, 3 * n:4 * n] = ends_y

            self.sigma_x.data[0, 0:n] = 0.2
            self.sigma_x.data[0, n:2 * n] = 1
            self.sigma_x.data[0, 2 * n:3 * n] = 0.2
            self.sigma_x.data[0, 3 * n:4 * n] = 1

            self.sigma_y.data[0, 0:n] = 1
            self.sigma_y.data[0, n:2 * n] = 0.2
            self.sigma_y.data[0, 2 * n:3 * n] = 1
            self.sigma_y.data[0, 3 * n:4 * n] = 0.2

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y):
        wave_num = 6 * torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_LPDI(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, x_wgs,y_wgs,sigma,bias: bool = True, device=None):
        super(PosEncoding_WPE_LPDI, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 100
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = nn.Parameter(torch.tensor(x_wgs, dtype=torch.float32))
        self.yc = nn.Parameter(torch.tensor(y_wgs, dtype=torch.float32))
        self.sigma_x = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.sigma_y = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y):
        wave_num = 6 * torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_LG(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WPE_LG, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 100
        self.weight_pe_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pe_l = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe_l = Parameter(torch.Tensor(out_features))
            self.bias_pe_g = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.1, 1)
        init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 0)
        nn.init.constant_(self.bias_pe_g, 0)
        nn.init.constant_(self.bias_pe_l, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe_g.to(device)
            self.weight_pe_l.to(device)
            self.weight_g.to(device)
            self.bias_pe_l.to(device)
            self.bias_pe_g.to(device)
            self.bias_g.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe_g, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pe_l, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe_g is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe_g, -bound, bound)

    def forward(self, x, y):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe_g = torch.matmul(xy_pe, self.weight_pe_g) + self.bias_pe_g

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = (torch.matmul(xy_pe, self.weight_pe_l) + self.bias_pe_l) * \
              (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + xy_pe_g

        return out

class PosEncoding_WPE_LGI(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WPE_LGI, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pe_l = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe_l = Parameter(torch.Tensor(out_features))
            self.bias_pe_g = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.1, 1)
        init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        start_x = -2 + 1 / 3  # 起始值 = -4/3
        ends_x = 2 - 1 / 3
        step_x = 5 / 6  # 间隔 1/3

        start_y = -1 + 1 / 6  # 起始值 = -4/3
        ends_y = 1 - 1 / 6
        step_y = 5 / 12  # 间隔 1/3
        n = 5

        with torch.no_grad():
            values_x = torch.arange(start_x, start_x + n * step_x, step_x)
            values_y = torch.arange(start_y, start_y + n * step_y, step_y)
            self.xc.data[0, 0:n] = start_x
            self.xc.data[0, n:2 * n] = values_x
            self.xc.data[0, 2 * n:3 * n] = ends_x
            self.xc.data[0, 3 * n:4 * n] = values_x

            self.yc.data[0, 0:n] = values_y
            self.yc.data[0, n:2 * n] = start_y
            self.yc.data[0, 2 * n:3 * n] = values_y
            self.yc.data[0, 3 * n:4 * n] = ends_y

            self.sigma_x.data[0, 0:n] = 0.2
            self.sigma_x.data[0, n:2 * n] = 1
            self.sigma_x.data[0, 2 * n:3 * n] = 0.2
            self.sigma_x.data[0, 3 * n:4 * n] = 1

            self.sigma_y.data[0, 0:n] = 1
            self.sigma_y.data[0, n:2 * n] = 0.2
            self.sigma_y.data[0, 2 * n:3 * n] = 1
            self.sigma_y.data[0, 3 * n:4 * n] = 0.2


            # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 0)
        nn.init.constant_(self.bias_pe_g, 0)
        nn.init.constant_(self.bias_pe_l, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe_g.to(device)
            self.weight_pe_l.to(device)
            self.weight_g.to(device)
            self.bias_pe_l.to(device)
            self.bias_pe_g.to(device)
            self.bias_g.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe_g, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pe_l, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe_g is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe_g, -bound, bound)

    def forward(self, x, y):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe_g = torch.matmul(xy_pe, self.weight_pe_g) + self.bias_pe_g

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = (torch.matmul(xy_pe, self.weight_pe_l) + self.bias_pe_l) * \
              (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + xy_pe_g

        return out

class PosEncoding_WPE_LGPDI(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, x_wgs,y_wgs,sigma,bias: bool = True, device = None):
        super(PosEncoding_WPE_LGPDI, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 100
        self.weight_pe_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pe_l = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = nn.Parameter(torch.tensor(x_wgs, dtype=torch.float32))
        self.yc = nn.Parameter(torch.tensor(y_wgs, dtype=torch.float32))
        self.sigma_x = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.sigma_y = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        # self.sigma = sigma.to(device)
        # self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim)).to(device)
        # self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim)).to(device)
        # init.uniform_(self.sigma_x, 1, 2)
        # init.uniform_(self.sigma_y, 1, 2)
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe_l = Parameter(torch.Tensor(out_features))
            self.bias_pe_g = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # init.uniform_(self.rhot, -1, 1)


            # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 0)
        nn.init.constant_(self.bias_pe_g, 0)
        nn.init.constant_(self.bias_pe_l, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe_g.to(device)
            self.weight_pe_l.to(device)
            self.weight_g.to(device)
            self.bias_pe_l.to(device)
            self.bias_pe_g.to(device)
            self.bias_g.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)
        # self.sigma_x = self.sigma_x * self.sigma
        # self.sigma_y = self.sigma_y * self.sigma
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe_g, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pe_l, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe_g is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe_g, -bound, bound)

    def forward(self, x, y):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe_g = torch.matmul(xy_pe, self.weight_pe_g) + self.bias_pe_g

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = (torch.matmul(xy_pe, self.weight_pe_l) + self.bias_pe_l) * \
              (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + xy_pe_g

        return out

class PosEncoding_WPE_LQ(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WPE_LQ, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 50
        self.weight_pe_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pe_l = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe_l = Parameter(torch.Tensor(out_features))
            self.bias_pe_g = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -1, 1)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.1, 1)
        init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_pe_g, 0)
        nn.init.constant_(self.bias_pe_l, 1)
        nn.init.constant_(self.bias, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe_g.to(device)
            self.weight_pe_l.to(device)
            self.weight_g.to(device)
            self.bias_pe_l.to(device)
            self.bias_pe_g.to(device)
            self.bias_g.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe_g, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        nn.init.constant_(self.weight_pe_l, 0)
        if self.bias_pe_g is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe_g, -bound, bound)


    def forward(self, x, y):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = (torch.matmul(xy_pe, self.weight_pe_l) + self.bias_pe_l) * \
              (torch.matmul(xy_pe, self.weight_pe_g) + self.bias_pe_g) * \
              (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias

        return out

class PosEncoding_WPE_PDI1(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int,x_wgs,y_wgs,bias: bool = True, device = None):
        super(PosEncoding_WPE_PDI1, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = nn.Parameter(torch.tensor(x_wgs, dtype=torch.float32))
        self.yc = nn.Parameter(torch.tensor(y_wgs, dtype=torch.float32))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_PDI(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int,x_wgs,y_wgs,sigma,bias: bool = True, device = None):
        super(PosEncoding_WPE_PDI, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = nn.Parameter(torch.tensor(x_wgs, dtype=torch.float32))
        self.yc = nn.Parameter(torch.tensor(y_wgs, dtype=torch.float32))
        self.sigma_x = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.sigma_y = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # init.uniform_(self.xc, -2, 2)
        # init.uniform_(self.yc, -1, 1)
        # init.uniform_(self.sigma_x, 0.2, 2)
        # init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_wgs3(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int,x_wgs,y_wgs,sigma,bias: bool = True, device = None):
        super(PosEncoding_WPE_wgs3, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = nn.Parameter(torch.tensor(x_wgs, dtype=torch.float32))
        self.yc = nn.Parameter(torch.tensor(y_wgs, dtype=torch.float32))
        self.sigma_x = nn.Parameter(torch.tensor(sigma*4, dtype=torch.float32))
        self.sigma_y = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # init.uniform_(self.xc, -2, 2)
        # init.uniform_(self.yc, -1, 1)
        # init.uniform_(self.sigma_x, 0.2, 2)
        # init.uniform_(self.sigma_y, 0.1, 1)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_wgs_rhot(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int,x_wgs,y_wgs,bias: bool = True, device = None):
        super(PosEncoding_WPE_wgs_rhot, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 20
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = nn.Parameter(torch.tensor(x_wgs, dtype=torch.float32))
        self.yc = nn.Parameter(torch.tensor(y_wgs, dtype=torch.float32))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)
        init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
                        x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_rhot(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WPE_rhot, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 20
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)
        init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
                        x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WPE_no_lowf(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WPE_no_lowf, self).__init__()
        self.num_freq = num_freq
        self.in_features = 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)
        init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe.to(device)
            self.weight_g.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_pe is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 27
        sinx = torch.sin(x * torch.pi * (wave_num ))
        cosx = torch.cos(x * torch.pi * (wave_num ))
        siny = torch.sin(y * torch.pi * (wave_num ))
        cosy = torch.cos(y * torch.pi * (wave_num ))
        xy_pe = torch.cat((sinx, cosx, siny, cosy), axis=-1)
        for i in range(1,self.num_freq):
            sinx = torch.sin(x * torch.pi * (wave_num - 5 * i))
            cosx = torch.cos(x * torch.pi * (wave_num - 5 * i))
            siny = torch.sin(y * torch.pi * (wave_num - 5 * i))
            cosy = torch.cos(y * torch.pi * (wave_num - 5 * i))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = torch.matmul(xy_pe, self.weight_pe) + self.bias_pe

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
                        x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_QWPE(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_QWPE, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe1 = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pe2 = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g1 = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.weight_g2 = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_pe1 = Parameter(torch.Tensor(out_features))
            self.bias_pe2 = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_g1 = Parameter(torch.Tensor(out_features))
            self.bias_g2 = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_g2, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        # nn.init.constant_(self.bias_pe1, 0)
        nn.init.constant_(self.bias_pe2, 1)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe1.to(device)
            self.weight_pe2.to(device)
            self.weight_g1.to(device)
            self.weight_g2.to(device)
            self.bias_pe.to(device)
            self.bias_pe1.to(device)
            self.bias_pe2.to(device)
            self.bias_g.to(device)
            self.bias_g1.to(device)
            self.bias_g2.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g1, a=math.sqrt(5))
        nn.init.constant_(self.weight_pe2, 0)
        nn.init.constant_(self.weight_g2, 0)
        if self.bias_pe1 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe1, -bound, bound)
        if self.bias_g1 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_g1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_g1, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 6*torch.pi

        xy_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq + 1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq + 1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq + 1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq + 1) * wave_num))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)
        xy_pe = (torch.matmul(xy_pe, self.weight_pe1) + self.bias_pe1) * (torch.matmul(xy_pe, self.weight_pe2) + self.bias_pe2) + self.bias_pe

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))
        # xy_gaus = (torch.matmul(xy_gaus, self.weight_g1) + self.bias_g1) * (torch.matmul(xy_gaus, self.weight_g2) + self.bias_g2) + self.bias_g
        xy_gaus = torch.matmul(xy_gaus, self.weight_g1) + self.bias_g1

        out = xy_pe * xy_gaus + self.bias_b

        return out

class PosEncoding_QWPE_no_lowf(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_QWPE_no_lowf, self).__init__()
        self.num_freq = num_freq
        self.in_features = 4 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pe1 = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pe2 = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g1 = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.weight_g2 = Parameter(torch.tensor(1,dtype=torch.float32))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_pe1 = Parameter(torch.Tensor(out_features))
            self.bias_pe2 = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_g1 = Parameter(torch.Tensor(out_features))
            self.bias_g2 = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_g2, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        # nn.init.constant_(self.bias_pe1, 0)
        nn.init.constant_(self.bias_pe2, 1)
        self.reset_parameters()

        if use_gpu:
            self.weight_pe1.to(device)
            self.weight_pe2.to(device)
            self.weight_g1.to(device)
            self.weight_g2.to(device)
            self.bias_pe.to(device)
            self.bias_pe1.to(device)
            self.bias_pe2.to(device)
            self.bias_g.to(device)
            self.bias_g1.to(device)
            self.bias_g2.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_pe1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g1, a=math.sqrt(5))
        nn.init.constant_(self.weight_pe2, 0)
        # nn.init.constant_(self.weight_g2, 0)
        if self.bias_pe1 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pe1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pe1, -bound, bound)
        if self.bias_g1 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_g1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_g1, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):

        wave_num = 27
        sinx = torch.sin(x * torch.pi * (wave_num ))
        cosx = torch.cos(x * torch.pi * (wave_num ))
        siny = torch.sin(y * torch.pi * (wave_num ))
        cosy = torch.cos(y * torch.pi * (wave_num ))
        xy_pe = torch.cat((sinx, cosx, siny, cosy), axis=-1)
        for i in range(1,self.num_freq):
            sinx = torch.sin(x * torch.pi * (wave_num - 5 * i))
            cosx = torch.cos(x * torch.pi * (wave_num - 5 * i))
            siny = torch.sin(y * torch.pi * (wave_num - 5 * i))
            cosy = torch.cos(y * torch.pi * (wave_num - 5 * i))
            xy_pe = torch.cat((xy_pe, sinx, cosx, siny, cosy), axis=-1)

        xy_pe = (torch.matmul(xy_pe, self.weight_pe1) + self.bias_pe1) * (torch.matmul(xy_pe, self.weight_pe2) + self.bias_pe2) + self.bias_pe

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))
        # xy_gaus = (torch.matmul(xy_gaus, self.weight_g1) + self.bias_g1) * (torch.matmul(xy_gaus, self.weight_g2) + self.bias_g2) + self.bias_g
        xy_gaus = torch.matmul(xy_gaus, (self.weight_g1 + self.weight_g2 * torch.eye(self.gaus_dim,self.out_features))) + self.bias_g1

        out = xy_pe * xy_gaus + self.bias_b

        return out

class PosEncoding_WQPE(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WQPE, self).__init__()
        self.num_freq = num_freq
        self.in_features = 1 + 2 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pex = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pey = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pex = Parameter(torch.Tensor(out_features))
            self.bias_pey = Parameter(torch.Tensor(out_features))
            self.bias_pe = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if use_gpu:
            self.weight_pex.to(device)
            self.weight_pey.to(device)
            self.weight_g.to(device)
            self.bias_pex.to(device)
            self.bias_pey.to(device)
            self.bias_pe.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)

    def reset_parameters(self) -> None:
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)

        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        nn.init.constant_(self.bias_pe, 0)
        # nn.init.constant_(self.weight_g, 0)
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pex, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pey, a=math.sqrt(5))

        if self.bias_pex is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pex)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pex, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pey)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pey, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 6*torch.pi

        x_pe = x
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq+1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq+1) * wave_num))
            x_pe = torch.cat((x_pe, sinx, cosx), axis=-1)


        y_pe = y
        for i in range(self.num_freq):
            siny = torch.sin(y * (2 ** (i - self.num_freq+1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq+1) * wave_num))
            y_pe = torch.cat((y_pe, siny, cosy), axis=-1)

        xy_pe = (torch.matmul(x_pe, self.weight_pex) + self.bias_pex) * (torch.matmul(y_pe, self.weight_pey) + self.bias_pey) + self.bias_pe

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()

        out = xy_pe * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g) + self.bias_b

        return out

class PosEncoding_WQPE_XY(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WQPE_XY, self).__init__()
        self.num_freq = num_freq
        self.in_features = 1 + 2 * num_freq
        self.out_features = out_features
        self.gaus_dim = 80
        self.weight_pex = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_pey = Parameter(torch.Tensor(self.in_features, out_features))

        self.weight_g_X = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc_X = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc_X = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x_X = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y_X = Parameter(torch.Tensor(1, self.gaus_dim))

        self.weight_g_Y = Parameter(torch.Tensor(self.gaus_dim, out_features))
        self.xc_Y = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc_Y = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x_Y = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y_Y = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_pex = Parameter(torch.Tensor(out_features))
            self.bias_pey = Parameter(torch.Tensor(out_features))
            self.bias_g_X = Parameter(torch.Tensor(out_features))
            self.bias_g_Y = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if use_gpu:
            self.weight_pex.to(device)
            self.weight_pey.to(device)
            self.weight_g_X.to(device)
            self.weight_g_Y.to(device)
            self.bias_pex.to(device)
            self.bias_pey.to(device)
            self.bias_g_X.to(device)
            self.bias_g_Y.to(device)
            self.bias_b.to(device)
            self.xc_X.to(device)
            self.yc_X.to(device)
            self.sigma_x_X.to(device)
            self.sigma_y_X.to(device)
            self.xc_Y.to(device)
            self.yc_Y.to(device)
            self.sigma_x_Y.to(device)
            self.sigma_y_Y.to(device)

    def reset_parameters(self) -> None:
        init.uniform_(self.xc_X, -2, 2)
        init.uniform_(self.yc_X, -1, 1)
        init.uniform_(self.sigma_x_X, 0.2, 2)
        init.uniform_(self.sigma_y_X, 0.1, 1)

        init.uniform_(self.xc_Y, -2, 2)
        init.uniform_(self.yc_Y, -1, 1)
        init.uniform_(self.sigma_x_Y, 0.2, 2)
        init.uniform_(self.sigma_y_Y, 0.1, 1)

        nn.init.constant_(self.bias_g_X, 1)
        nn.init.constant_(self.bias_g_Y, 1)
        nn.init.constant_(self.bias_b, 0)
        # nn.init.constant_(self.weight_g, 0)
        init.kaiming_uniform_(self.weight_g_X, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g_Y, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pex, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_pey, a=math.sqrt(5))

        if self.bias_pex is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pex)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pex, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_pey)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_pey, -bound, bound)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 6*torch.pi

        x_pe = x
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq+1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq+1) * wave_num))
            x_pe = torch.cat((x_pe, sinx, cosx), axis=-1)


        y_pe = y
        for i in range(self.num_freq):
            siny = torch.sin(y * (2 ** (i - self.num_freq+1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq+1) * wave_num))
            y_pe = torch.cat((y_pe, siny, cosy), axis=-1)

        x_gaus = torch.exp(-(((x - self.xc_X) / self.sigma_x_X) ** 2 + ((y - self.yc_X) / self.sigma_y_X) ** 2))
        y_gaus = torch.exp(-(((x - self.xc_Y) / self.sigma_x_Y) ** 2 + ((y - self.yc_Y) / self.sigma_y_Y) ** 2))

        x_pe = (torch.matmul(x_pe, self.weight_pex) + self.bias_pex) * (
               torch.matmul(x_gaus, self.weight_g_X) + self.bias_g_X)
        y_pe = (torch.matmul(y_pe, self.weight_pey) + self.bias_pey) * (
               torch.matmul(y_gaus, self.weight_g_Y) + self.bias_g_Y)

        out = x_pe + y_pe + self.bias_b

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        return out

class PosEncoding_WPE_high(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_freq: int, bias: bool = True, device = None):
        super(PosEncoding_WPE_high, self).__init__()
        self.num_freq = num_freq
        self.in_features = in_features + 2 * num_freq * in_features
        self.out_features = out_features
        self.num_freq_high = 3
        self.gaus_dim = 80
        self.num_local = in_features + 2 * self.num_freq_high * in_features
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, self.num_local))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_g = Parameter(torch.Tensor(self.num_local))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.2, 2)
        init.uniform_(self.sigma_y, 0.1, 1)

        init.uniform_(self.weight_g, -1, 1)
        nn.init.constant_(self.bias_g, 0)

        if use_gpu:
            self.weight_g.to(device)
            self.bias_g.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 6*torch.pi
        wave_num_high = 16*torch.pi
        # sinx = torch.sin(x * (2 ** (- 5) * wave_num))
        # cosx = torch.cos(x * (2 ** (- 5) * wave_num))
        # x_pe = torch.cat((x, y, sinx, cosx), axis=-1)

        x_pe = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - self.num_freq+1) * wave_num))
            cosx = torch.cos(x * (2 ** (i - self.num_freq+1) * wave_num))
            siny = torch.sin(y * (2 ** (i - self.num_freq+1) * wave_num))
            cosy = torch.cos(y * (2 ** (i - self.num_freq+1) * wave_num))
            x_pe = torch.cat((x_pe, sinx, cosx, siny, cosy), axis=-1)
        gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2))
        gaus = torch.matmul(gaus, self.weight_g) + self.bias_g
        x_pe_high = torch.cat((x, y), axis=-1)
        for i in range(self.num_freq_high):
            sinx = torch.sin(x * (2 ** i * wave_num_high))
            cosx = torch.cos(x * (2 ** i * wave_num_high))
            siny = torch.sin(y * (2 ** i * wave_num_high))
            cosy = torch.cos(y * (2 ** i * wave_num_high))
            x_pe_high = torch.cat((x_pe_high, sinx, cosx, siny, cosy), axis=-1)
        x_pe_local = x_pe_high * gaus
        out = torch.cat((x_pe, x_pe_local), axis=-1)
        return out

class wavelet_act(nn.Module):
    def __init__(self, in_features: int, gaus_dim: int, bias: bool = True, device = None):
        super(wavelet_act, self).__init__()
        self.gaus_dim = gaus_dim
        self.weight_g = Parameter(torch.Tensor(self.gaus_dim, in_features))
        self.xc = Parameter(torch.Tensor(1,self.gaus_dim))
        self.yc = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_x = Parameter(torch.Tensor(1, self.gaus_dim))
        self.sigma_y = Parameter(torch.Tensor(1, self.gaus_dim))
        # self.rhot = Parameter(torch.Tensor(1, self.gaus_dim))
        if bias:
            self.bias_g = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        init.uniform_(self.xc, -2, 2)
        init.uniform_(self.yc, -1, 1)
        init.uniform_(self.sigma_x, 0.1, 0.5)
        init.uniform_(self.sigma_y, 0.1, 0.5)
        # init.uniform_(self.rhot, -1, 1)

        # nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.bias_g, 1)
        self.reset_parameters()

        if use_gpu:
            self.weight_g.to(device)
            self.bias_g.to(device)
            self.xc.to(device)
            self.yc.to(device)
            self.sigma_x.to(device)
            self.sigma_y.to(device)
            # self.rhot.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))

    def forward(self, x, y, z):

        # xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 - 2 * self.rhot * (
        #                 x - self.xc) * (y - self.yc) / (self.sigma_x * self.sigma_y)))

        xy_gaus = torch.exp(-(((x - self.xc) / self.sigma_x) ** 2 + ((y - self.yc) / self.sigma_y) ** 2 ))

        # self.gaus = (torch.matmul(xy_gaus, self.weight_g) + self.bias_g).detach()
        out = z * (torch.matmul(xy_gaus, self.weight_g) + self.bias_g)

        return out

if __name__ == '__main__':
    a = torch.randn(10, 20)
    b = PosEncoding_trainable(20, 128)
    c = b(a)
    print(c.shape)
    print(c)