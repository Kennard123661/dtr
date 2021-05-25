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


def get_normalization(normalization: str, num_dims: int):
    assert normalization is not None
    if normalization == 'batch':
        return nn.BatchNorm1d(num_features=num_dims)
    elif normalization == 'instance':
        return nn.InstanceNorm1d(num_features=num_dims, affine=True)
    elif normalization == 'layer':
        return nn.LayerNorm(num_dims)
    else:
        raise ValueError('no such normalization: {}'.format(normalization))


class MlpNet(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, base_ndims: int, nlayers: int, normalization: str = None,
                 activation: str = None):
        """
        Helper class for Mlp Net made of more than 1 linear layer.

        Args:
            in_ndims: number of input feature dimensions
            out_ndims: number of output feature dimensions
            base_ndims: number of intermediate feature dimensions
            nlayers: number of linear layers
            normalization: boolean variable indicating whether there is batch normalization.
            activation: activations for the mlp layers.
        """
        super(MlpNet, self).__init__()
        assert nlayers >= 2, 'use Linear instead of MlpNet for single layers'

        net = [Linear(in_ndims=in_ndims, out_ndims=base_ndims, activation=activation, normalization=normalization)]
        for _ in range(nlayers - 2):
            net.append(Linear(in_ndims=base_ndims, out_ndims=base_ndims, activation=activation,
                              normalization=normalization))
        net.append(Linear(in_ndims=base_ndims, out_ndims=out_ndims, activation=activation, normalization=normalization))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class Linear(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, activation: str = None, normalization: str = None):
        """
        Helper for linear layer

        Args:
            in_ndims: number of input feature dimensions
            out_ndims: number of output feature diemnsions
            activation: activation the layer
            normalization: whether there is batch norm.
        """
        super(Linear, self).__init__()
        net = [nn.Linear(in_features=in_ndims, out_features=out_ndims)]
        if activation is not None:
            net.append(get_activation(activation=activation))
        if normalization == 'instance':
            normalization = 'layer'

        if normalization is not None:
            net.append(get_normalization(normalization=normalization, num_dims=out_ndims))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class Conv1d(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, ksize: int, padding: int = 0,
                 activation: str = None, normalization: str = None):
        """
        Helper for Conv1d layer

        Args:
            in_ndims: number of input dimensions
            out_ndims: number of output dimensions
            ksize: kernel size
            padding: number of padding
            activation: activation name
            normalization: boolean indicating whether there is batch norm
        """
        super(Conv1d, self).__init__()
        net = [nn.Conv1d(in_channels=in_ndims, out_channels=out_ndims, kernel_size=ksize, padding=padding)]
        if activation is not None:
            net.append(get_activation(activation=activation))
        if normalization is not None:
            net.append(get_normalization(normalization=normalization, num_dims=out_ndims))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class Conv1dNet(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, base_ndims: int, nlayers: int, ksize: int, padding: int,
                 activation: str = None, normalization: str = None):
        super(Conv1dNet, self).__init__()
        assert nlayers >= 2, 'Use Conv1d for single layer.'

        net = [Conv1d(in_ndims=in_ndims, out_ndims=base_ndims, ksize=ksize, padding=padding,
                      activation=activation, normalization=normalization)]
        for _ in range(nlayers - 2):
            net.append(Conv1d(in_ndims=base_ndims, out_ndims=base_ndims, ksize=ksize,
                              padding=padding, activation=activation, normalization=normalization))
        net.append(Conv1d(in_ndims=base_ndims, out_ndims=out_ndims, ksize=ksize,
                          padding=padding, activation=activation, normalization=normalization))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out

