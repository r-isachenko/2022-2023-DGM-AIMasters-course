from .visualize import init_samples_vis_ctx, show_epoch_samples

from collections import defaultdict
from tqdm.notebook import tqdm

import torch
from torch import optim


def train_epoch(model, train_loader, optimizer, use_cuda, loss_key='total'):
    model.train()

    stats = defaultdict(list)
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


def eval_model(model, data_loader, use_cuda):
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
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    use_tqdm=False,
    use_cuda=False,
    visualization_samples_enabled=True, # model should have 'sample' method
    sample_kwargs={},
    loss_key='total_loss',
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.cuda()
    if visualization_samples_enabled:
        samples_ctx = init_samples_vis_ctx()

    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, loss_key)
        test_loss = eval_model(model, test_loader, use_cuda)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])

        # visualise samples
        if visualization_samples_enabled:
            with torch.no_grad():
                model.eval()
                samples = model.sample(64, **sample_kwargs)
                if torch.is_tensor(samples):
                    samples = samples.detach().cpu()
                show_epoch_samples(samples_ctx, samples, title=f'Samples (epoch={epoch})')

    return dict(train_losses), dict(test_losses)
