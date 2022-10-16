import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import matplotlib
import torch.autograd as autograd

import sys
sys.path.append('../../homeworks')
from dgm_utils.visualize import (
    TICKS_FONT_SIZE, 
    LEGEND_FONT_SIZE, 
    LABEL_FONT_SIZE,
    TITLE_FONT_SIZE)

def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

def visualize_2d_map(
    data, mapped_data, title,
    data_color='red', mapped_data_color='blue', map_color='green',
    data_label=None, mapped_data_label=None, map_label=None,
    xlabel=None, ylabel=None,
    s=1, linewidth=0.2, map_alpha=0.6, data_alpha=0.4,
    figsize=(5, 5), dpi=None):
    
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()
    data = make_numpy(data)
    mapped_data = make_numpy(mapped_data)
    lines = np.concatenate([data, mapped_data], axis=-1).reshape(-1, 2, 2)
    lc = matplotlib.collections.LineCollection(
        lines, color=map_color, linewidths=linewidth, alpha=map_alpha, label=map_label)
    ax.add_collection(lc)
    ax.scatter(
        data[:, 0], data[:, 1], s=s, label=data_label,
        alpha=data_alpha, zorder=2, color=data_color)
    ax.scatter(
        mapped_data[:, 0], mapped_data[:, 1], s=s, label=mapped_data_label,
        alpha=data_alpha, zorder=2, color=mapped_data_color)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.show()
    
def visualize_samples_pdf(samples, samples_pdf,
    title, n_spaces=400, color_grads='plasma', 
    log_scale=False, ax=None, s=0.5, alpha=1.,
    figsize=(5, 5), dpi=None):
    
    def normalize_data(X, v_min=0., v_max=1.):
        X += (v_min - X.min())
        if X.max() > 0.:
            X *= (v_max / X.max())
        return X
    
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()
    probs = make_numpy(samples_pdf)
    samples = make_numpy(samples)
    
    if log_scale:
        probs = np.log(samples_pdf)
    colors = matplotlib.colormaps[color_grads](normalize_data(probs))
    cax = ax.scatter(samples[:,0], samples[:,1], c=colors, s=s, alpha=alpha)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.show()

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

def manual_zero_grad(*tensors):
    for tsr in tensors:
        if tsr.grad is not None:
            tsr.grad.detach_()
            tsr.grad.zero_()
    