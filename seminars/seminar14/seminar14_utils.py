import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as TD
from torch import optim

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import matplotlib
import torch.autograd as autograd

from collections import defaultdict
from tqdm.notebook import tqdm

import sys
sys.path.append('../../homeworks')

class FullyConnectedMLP(nn.Module):

    def __init__(
        self, 
        input_dim, 
        hiddens, 
        output_dim, 
        activation_gen = lambda : nn.ReLU()
    ):

        assert isinstance(hiddens, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens

        model = []
        prev_h = input_dim
        for h in hiddens:
            model.append(nn.Linear(prev_h, h))
            model.append(activation_gen())
            prev_h = h
        model.append(nn.Linear(hiddens[-1], output_dim))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, self.output_dim)

def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

def batch_jacobian(input, output, create_graph=True, retain_graph=True):
    '''
    :Parameters:
    input : tensor (bs, *shape_inp)
    output: tensor (bs, *shape_oup) , NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, *shape_oup, *shape_inp)
    '''
    def out_permutation():
        n_inp = np.arange(len(input.shape) - 1)
        n_output = np.arange(len(output.shape) - 1)
        return tuple(np.concatenate([n_output + 1, [0,], n_inp + len(n_output) + 1]))

    s_output = torch.sum(output, dim=0) # sum by batch dimension
    batched_grad_outputs = torch.eye(
        np.prod(s_output.shape)).view((-1,) + s_output.shape).to(output)
    # batched_grad_outputs = torch.eye(s_output.size(0)).to(output)
    grad = autograd.grad(
        outputs=s_output, inputs=input,
        grad_outputs=batched_grad_outputs,
        create_graph=create_graph, 
        retain_graph=retain_graph,
        only_inputs=True,
        is_grads_batched=True
    )
    return grad[0].permute(out_permutation())

def plot_score_function(
    score_function, 
    data, 
    title, 
    plot_scatter=True, 
    xlim=(-2., 2.), 
    ylim=(-2., 2.), 
    npts=40, 
    ax=None, 
    figsize=(12, 12),
    scatter_label='GT labels',
    quiver_label = None,
    quiver_color='black',
    device='cuda'
):
    xx = np.stack(
        np.meshgrid(
            np.linspace(xlim[0], xlim[1], npts), 
            np.linspace(ylim[0], ylim[1], npts)), axis=-1).reshape(-1, 2)
    scores = score_function(torch.tensor(xx).float().to(device)).detach().cpu().numpy()
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # Perform the plots
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if (plot_scatter):
        np_data = make_numpy(data)
        ax.scatter(
            np_data[:, 0], np_data[:, 1], alpha=0.3, 
            color='red', edgecolor='white', s=40, label=scatter_label)
    ax.quiver(*xx.T, *scores_log1p.T, width=0.002, color=quiver_color, label=quiver_label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=16)
    return ax