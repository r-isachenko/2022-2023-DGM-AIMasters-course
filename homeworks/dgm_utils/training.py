from .visualize import init_visual_ctx, show_epoch_samples_losses

from collections import defaultdict
from tqdm.notebook import tqdm

import torch
import numpy as np
from torch import optim


def train_epoch(
    model: object,
    train_loader: object,
    optimizer: object,
    use_cuda: bool,
    loss_key: str = "total",
) -> defaultdict:
    model.train()

    stats = defaultdict(list)
    loss_key = kwargs['loss_key']
    for x in train_loader:
        if use_cuda:
            x = x.cuda()
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(model: object, data_loader: object, use_cuda: bool) -> defaultdict:
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for x in data_loader:
            if use_cuda:
                x = x.cuda()
            losses = model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset)
    return stats


def train_model(
    model: object,
    train_loader: object,
    test_loader: object,
    epochs: int,
    lr: float,
    use_tqdm: bool = False,
    use_cuda: bool = False,
    loss_key: str = "total_loss",
) -> Tuple[dict, dict]:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    
    if use_cuda:
        model = model.cuda()

    if visualization_enabled:
        assert hasattr(model, "sample")
        ctx = init_visual_ctx()

    for epoch in forrange:
        model.train()
        
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, **train_params_dict)
        test_loss = eval_model(model, test_loader, use_cuda)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])

        if visualization_enabled:
            with torch.no_grad():
                model.eval()
                samples = model.sample(64, **sample_kwargs)
                if torch.is_tensor(samples):
                    samples = samples.cpu()
                show_epoch_samples_losses(ctx, {
                    'samples' : samples, 
                    'title': f'Samples (epoch={epoch})'
                }, 
                {
                    'train_losses' : train_losses,
                    'test_losses' : test_losses
                })

    return dict(train_losses), dict(test_losses)
