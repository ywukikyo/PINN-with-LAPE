import torch
import math
import torch.nn as nn
from torch.nn import Parameter, init

use_gpu = torch.cuda.is_available()


class Quadratic_operation(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Quadratic_operation, self).__init__()
        self.in_features = int(in_features * (in_features + 1) / 2 + in_features)
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        init.constant_(self.bias, 0)
        init.xavier_uniform_(self.weight, gain=1)
    def forward(self, x):
        if use_gpu:
            self.weight.cuda()
            self.bias.cuda()
        XC = x
        for ih in range(0, x.shape[1]):
            for jh in range(0, x.shape[1] - ih):
                XC = torch.cat([XC, torch.unsqueeze(XC[:, ih] * XC[:, ih + jh],1)], 1)
        out = torch.matmul(XC, self.weight)+ self.bias
        return out

class Square_operation(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Square_operation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        init.constant_(self.bias, 0)
        init.xavier_uniform_(self.weight, gain=1)
    def forward(self, x):
        if use_gpu:
            self.weight.cuda()
            self.bias.cuda()

        out = (torch.matmul(x, self.weight)+ self.bias)**2
        return out

class Quadratic_product_operation_0(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super(Quadratic_product_operation_0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(in_features, out_features))
        self.weight_b = Parameter(torch.Tensor(in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.weight_b.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x):

        out = (torch.matmul(x, self.weight_r) + self.bias_r)*(torch.matmul(x, self.weight_g) + self.bias_g)\
              + torch.matmul(torch.pow(x, 2), self.weight_b) + self.bias_b
        return out

class Quadratic_product_operation_local(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None):
        super(Quadratic_product_operation_local, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(in_features, out_features))
        self.weight_b = Parameter(torch.Tensor(in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.weight_b.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self,a,b, x):

        out = (torch.matmul(x, self.weight_r) + self.bias_r)*(torch.matmul(x, self.weight_g) + self.bias_g)\
              + torch.matmul(torch.pow(x, 2), self.weight_b) + self.bias_b
        return out

class Quadratic_product_operation(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None):
        super(Quadratic_product_operation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(in_features, out_features))
        self.weight_b = Parameter(torch.Tensor(in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
            self.bias_t = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 1)
        nn.init.constant_(self.bias_t, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.weight_b.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.bias_t.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x):

        out = (torch.matmul(x, self.weight_r) + self.bias_r)*(torch.matmul(x, self.weight_g) + self.bias_g)*(torch.matmul(x, self.weight_b) + self.bias_b) + self.bias_t
        return out

class Quadratic_positional_encoding(nn.Module):
    def __init__(self, in_features: int, out_features: int,num_freq: int,bias: bool = True, device = None):
        super(Quadratic_positional_encoding, self).__init__()
        self.f = 10
        self.num_freq = num_freq
        self.in_features = in_features + self.num_freq * 4 + 2
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_b = Parameter(torch.Tensor(self.in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.weight_b.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x, y, v_gaus, v0_gaus):
        wave_num = 2*torch.pi * self.f / v_gaus
        wave_num0 = 2 * torch.pi * self.f / v0_gaus

        sinx = torch.sin(x * (2 ** (- 4) * wave_num0))
        cosx = torch.cos(x * (2 ** (- 4) * wave_num0))
        x_pe0 = torch.cat((x, y, sinx, cosx), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - 3) * wave_num0))
            cosx = torch.cos(x * (2 ** (i - 3) * wave_num0))
            siny = torch.sin(y * (2 ** (i - 3) * wave_num0))
            cosy = torch.cos(y * (2 ** (i - 3) * wave_num0))
            x_pe0 = torch.cat((x_pe0, sinx, cosx, siny, cosy), axis=-1)

        sinx = torch.sin(x * (2 ** (- 4) * wave_num))
        cosx = torch.cos(x * (2 ** (- 4) * wave_num))
        x_pe = torch.cat((x, y, sinx, cosx), axis=-1)
        for i in range(self.num_freq):
            sinx = torch.sin(x * (2 ** (i - 3) * wave_num))
            cosx = torch.cos(x * (2 ** (i - 3) * wave_num))
            siny = torch.sin(y * (2 ** (i - 3) * wave_num))
            cosy = torch.cos(y * (2 ** (i - 3) * wave_num))
            x_pe = torch.cat((x_pe, sinx, cosx, siny, cosy), axis=-1)

        out = (torch.matmul(x_pe0, self.weight_r) + self.bias_r)*(torch.matmul(x_pe, self.weight_g) + self.bias_g)\
              + torch.matmul(torch.pow(x_pe, 2), self.weight_b) + self.bias_b
        return out

class Quadratic_positional_encoding_1(nn.Module):
    def __init__(self, in_features: int, out_features: int,num_freq: int,bias: bool = True, device = None):
        super(Quadratic_positional_encoding_1, self).__init__()
        self.f = 10
        self.num_freq = num_freq
        self.in_features = num_freq*2+1
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.in_features, out_features))
        # self.weight_b = Parameter(torch.Tensor(self.in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))

        else:
            self.register_parameter('bias', None)
        # nn.init.constant_(self.weight_g, 0)
        # nn.init.constant_(self.weight_b, 0)
        # nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_g, -bound, bound)



    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x, y, v_gaus, v0_gaus):
        # wave_num_gaus = 2 * torch.pi * self.f / v_gaus
        wave_num = 6 * torch.pi
        # wave_num0 = 2 * torch.pi * self.f / v0_gaus
        xy = torch.cat((x, y), axis=-1)
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

        out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (torch.matmul(y_pe, self.weight_g) + self.bias_g)  + self.bias_b
        # out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (
        #             torch.matmul(y_pe, self.weight_g) + self.bias_g) + wave_num_gaus * self.bias_b
        return out

class Quadratic_positional_encoding_2(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None):
        super(Quadratic_positional_encoding_2, self).__init__()
        self.in_features = in_features
        self.out_features = int(out_features/2)
        self.weight_r = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight_g = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight_b = Parameter(torch.Tensor(self.in_features, self.out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(self.out_features))
            self.bias_g = Parameter(torch.Tensor(self.out_features))
            self.bias_b = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.weight_b.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)


    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_r.uniform_(-5, 5, generator=None)
        self.__reset_bias()

    def forward(self, x):

        xy = (torch.matmul(x, self.weight_r) + self.bias_r)*(torch.matmul(x, self.weight_g) + self.bias_g)+ self.bias_b
        # xy = torch.matmul(x, self.weight_r) + self.bias_r
        sin = torch.sin(xy*torch.pi)
        cos = torch.cos(xy*torch.pi)
        out = torch.cat((sin, cos),axis=-1)
        return out

class Quadratic_positional_encoding_3(nn.Module):
    def __init__(self, in_features: int, out_features: int,num_freq: int,bias: bool = True, device = None):
        super(Quadratic_positional_encoding_3, self).__init__()
        self.f = 10
        self.num_freq = num_freq
        self.in_features = num_freq*2+1
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_xy1 = Parameter(torch.Tensor(in_features, out_features))
        self.weight_xy2 = Parameter(torch.Tensor(in_features, out_features))
        # self.weight_b = Parameter(torch.Tensor(self.in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
            self.bias_xy1 = Parameter(torch.Tensor(out_features))
            self.bias_xy2 = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # nn.init.constant_(self.weight_g, 0)
        # nn.init.constant_(self.weight_b, 0)
        # nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.weight_xy1.to(device)
            self.weight_xy2.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.bias_xy1.to(device)
            self.bias_xy2.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_g, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_xy1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_xy1, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_xy2)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_xy2, -bound, bound)



    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        # init.kaiming_uniform_(self.weight_xy1, a=math.sqrt(5))
        # init.kaiming_uniform_(self.weight_xy2, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_xy1.uniform_(-2, 2, generator=None)
            self.weight_xy2.uniform_(-2, 2, generator=None)
        self.__reset_bias()

    def forward(self, x, y, v_gaus, v0_gaus):
        # wave_num_gaus = 2 * torch.pi * self.f / v_gaus
        wave_num = 8 * torch.pi
        # wave_num0 = 2 * torch.pi * self.f / v0_gaus
        xy = torch.cat((x, y), axis=-1)
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

        out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (
                    torch.matmul(y_pe, self.weight_g) + self.bias_g) * torch.exp(-(
                    (torch.matmul(xy, self.weight_xy1) + self.bias_xy1) * (
                        torch.matmul(xy, self.weight_xy2) + self.bias_xy2))**2) + self.bias_b
        # out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (
        #             torch.matmul(y_pe, self.weight_g) + self.bias_g) + wave_num_gaus * self.bias_b
        return out

class Quadratic_positional_encoding_4(nn.Module):
    def __init__(self, in_features: int, out_features: int,num_freq: int,bias: bool = True, device = None):
        super(Quadratic_positional_encoding_4, self).__init__()
        self.f = 10
        self.num_freq = num_freq
        self.in_features = num_freq*2+1
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.device = device
        # self.weight_b = Parameter(torch.Tensor(self.in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))

        else:
            self.register_parameter('bias', None)
        # nn.init.constant_(self.weight_g, 0)
        # nn.init.constant_(self.weight_b, 0)
        # nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_g, -bound, bound)



    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x, y, v_gaus, v0_gaus):
        # wave_num_gaus = 2 * torch.pi * self.f / v_gaus
        k = 14.0
        k_step = torch.linspace(1/(2*k), 1, steps=self.num_freq).to(self.device)
        wave_num = k * torch.pi
        # wave_num0 = 2 * torch.pi * self.f / v0_gaus
        # xy = torch.cat((x, y), axis=-1)
        x_pe = x
        for i in range(self.num_freq):
            sinx = torch.sin(x * k_step[i] * wave_num)
            cosx = torch.cos(x * k_step[i] * wave_num)
            x_pe = torch.cat((x_pe, sinx, cosx), axis=-1)


        y_pe = y
        for i in range(self.num_freq):
            siny = torch.sin(y * k_step[i] * wave_num)
            cosy = torch.cos(y * k_step[i] * wave_num)
            y_pe = torch.cat((y_pe, siny, cosy), axis=-1)

        out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (torch.matmul(y_pe, self.weight_g) + self.bias_g)  + self.bias_b
        # out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (
        #             torch.matmul(y_pe, self.weight_g) + self.bias_g) + wave_num_gaus * self.bias_b
        return out

class Quadratic_positional_encoding_5(nn.Module):
    def __init__(self, in_features: int, out_features: int,num_freq: int,bias: bool = True, device = None):
        super(Quadratic_positional_encoding_5, self).__init__()
        self.f = 20
        self.num_freq = num_freq
        self.in_features = num_freq*2+1
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_g = Parameter(torch.Tensor(self.in_features, out_features))
        self.weight_xy1 = Parameter(torch.Tensor(in_features, out_features))
        self.weight_xy2 = Parameter(torch.Tensor(in_features, out_features))
        # self.weight_b = Parameter(torch.Tensor(self.in_features, out_features))



        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
            self.bias_xy1 = Parameter(torch.Tensor(out_features))
            self.bias_xy2 = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # nn.init.constant_(self.weight_g, 0)
        # nn.init.constant_(self.weight_b, 0)
        # nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

        if use_gpu:
            self.weight_r.to(device)
            self.weight_g.to(device)
            self.weight_xy1.to(device)
            self.weight_xy2.to(device)
            self.bias_r.to(device)
            self.bias_g.to(device)
            self.bias_b.to(device)
            self.bias_xy1.to(device)
            self.bias_xy2.to(device)

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_g)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_g, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_xy1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_xy1, -bound, bound)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_xy2)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_xy2, -bound, bound)



    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        # init.kaiming_uniform_(self.weight_xy1, a=math.sqrt(5))
        # init.kaiming_uniform_(self.weight_xy2, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_xy1.uniform_(-2, 2, generator=None)
            self.weight_xy2.uniform_(-2, 2, generator=None)
        self.__reset_bias()

    def forward(self, x, y, v_gaus, v0_gaus):
        # wave_num_gaus = 2 * torch.pi * self.f / v_gaus
        wave_num = 8 * torch.pi
        # wave_num0 = 2 * torch.pi * self.f / v0_gaus
        xy = torch.cat((x, y), axis=-1)
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

        out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (
                    torch.matmul(y_pe, self.weight_g) + self.bias_g) * torch.exp(-(
                    (torch.matmul(xy, self.weight_xy1) + self.bias_xy1) * (
                        torch.matmul(xy, self.weight_xy2) + self.bias_xy2))**2) + self.bias_b
        # out = (torch.matmul(x_pe, self.weight_r) + self.bias_r) * (
        #             torch.matmul(y_pe, self.weight_g) + self.bias_g) + wave_num_gaus * self.bias_b
        return out

if __name__ == '__main__':
    a = torch.randn(10, 28*28)
    # b = Quadratic_operation(28*28, 128)
    b = Square_operation(28 * 28, 128)
    c = b(a)
    print(c.shape)
    print(c)