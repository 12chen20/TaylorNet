from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np


class Taylor_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Taylor_prop, self).__init__(aggr='add', **kwargs)#aggr所用的聚合方法
        # 当传入字典形式的参数时，就要使用 ** kwargs
        self.K = K
        # 定义新的初始化变量。模型中的参数，它是Parameter()类，
        # 先转化为张量，再转化为可训练的Parameter对象
        # Parameter用于将参数自动加入到参数列表
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)#Fills self tensor with the specified value.


    def forward(self, x, edge_index, edge_weight=None):

        TEMP = self.temp

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        # 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        #第一项
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # out = (1 / math.factorial(0)) * (1 / (2 ** 0)) * TEMP[0] * tmp[0]
        out = (1 / math.factorial(0)) * (1 / (2 ** 0)) * TEMP[0] * tmp[0]

        for i in range(self.K):
            x = tmp[ i + 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (-1)**(i + 1) *(1 / math.factorial(i + 1)*comb(2* (i + 1), i + 1)) * TEMP[i + 1] * x
            # out = out + (i + 3)*((-1)**(i + 1) / 2) * (1 / math.factorial(i + 1) * math.factorial(i + 1)) * (1 /comb(2* (i + 1), i + 1)) * TEMP[i + 1] * x

        return out
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
