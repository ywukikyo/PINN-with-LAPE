import torch
from collections import OrderedDict
import torch.nn as nn
import Quadratic_neuron as QN
import PosEncoding as PE


class Model(torch.nn.Module):
    def __init__(self, layers, PosEncoding, nettype, x_wgs,y_wgs,sigma,device):
        super(Model, self).__init__()

        # parameters
        self.depth = len(layers) - 1
        self.num_pe = 5
        # set up layer order dict
        self.activation = torch.nn.Tanh
        self.PosEncoding = PosEncoding

        layer_list = list()
        if self.PosEncoding == 'n':
            if nettype == 'DNN':
                layer_list.append(('layer_%d' % 0, torch.nn.Linear(layers[0], layers[1],)))
                layer_list.append(('activation_%d' % 0, self.activation()))
            elif nettype == 'QNN':
                layer_list.append(('layer_%d' % 0, QN.Quadratic_product_operation_0(layers[0], layers[1],device = device)))
                layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'pe_l':
            self.lay_pe = PE.PosEncoding_f(layers[0], self.num_pe)
            layer_list.append(('layer_%d' % 0, torch.nn.Linear(layers[0] * (2 * self.num_pe + 1), layers[1])))
            layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'pe_q':
            self.lay_pe = PE.PosEncoding_f(layers[0], self.num_pe)
            layer_list.append(('layer_%d' % 0, QN.Quadratic_product_operation_0(layers[0] * (2 * self.num_pe + 1), layers[1],device = device)))
            layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'lape_l':
            self.lay_pe = PE.PosEncoding_WPE_L(layers[0], layers[1], self.num_pe,device = device)
            # layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'lape_li':
            self.lay_pe = PE.PosEncoding_WPE_LI(layers[0], layers[1], self.num_pe,device = device)
            layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'lape_lg':
            self.lay_pe = PE.PosEncoding_WPE_LG(layers[0], layers[1], self.num_pe,device = device)
            layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'lape_lgi':
            self.lay_pe = PE.PosEncoding_WPE_LGI(layers[0], layers[1], self.num_pe,device = device)
            layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'lape_lq':
            self.lay_pe = PE.PosEncoding_WPE_LQ(layers[0], layers[1], self.num_pe,device = device)
            layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'lape_lpdi':
            self.lay_pe = PE.PosEncoding_WPE_LPDI(layers[0], layers[1], self.num_pe, x_wgs,y_wgs,sigma,device = device)
            layer_list.append(('activation_%d' % 0, self.activation()))
        elif self.PosEncoding == 'lape_lgpdi':
            self.lay_pe = PE.PosEncoding_WPE_LGPDI(layers[0], layers[1], self.num_pe, x_wgs,y_wgs,sigma,device = device)
            layer_list.append(('activation_%d' % 0, self.activation()))

        if nettype == 'DNN':
            for i in range(1, self.depth - 1):
                layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
                layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        elif nettype == 'QNN':
            for i in range(1, self.depth - 1):
                layer_list.append(('layer_%d' % i, QN.Quadratic_product_operation_0(layers[i], layers[i + 1],device = device)))
                layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))

        layerDict = OrderedDict(layer_list)
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)


    def forward(self, x, y):
        if self.PosEncoding == 'n':
            out = self.layers(torch.cat([x, y], dim=1))
        else:
            out = self.lay_pe(x, y)
            # self.gaus = self.lay_pe.gaus
            out = self.layers(out)

        return out

