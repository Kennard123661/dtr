import torch
import torch.nn as nn


def get_activation(activation: str):
    assert activation is not None
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'lrelu':
        return nn.LeakyReLU(inplace=False)
    elif activation == 'elu':
        return nn.ELU(inplace=False)
    else:
        raise NotImplementedError


class Linear(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, activation: str = None, batchnorm: bool = False):
        super(Linear, self).__init__()
        layers = [nn.Linear(in_features=in_dims, out_features=out_dims)]
        if activation is not None:
            layers.append(get_activation(activation=activation))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)


class Conv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0,
                 activation: str = None, batchnorm: bool = False):
        super(Conv1d, self).__init__()
        layers = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, padding=padding)]
        if activation is not None:
            layers.append(get_activation(activation=activation))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)
