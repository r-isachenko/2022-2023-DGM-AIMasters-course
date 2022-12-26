from .visualize import init_samples_vis_ctx, show_epoch_samples

from collections import defaultdict
from tqdm.notebook import tqdm

import np
import torch
from torch import optim


def train_epoch(model, train_loader, optimizer, use_cuda, **kwargs):
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


def train_epoch_vae_kl_anneal(model, train_loader, optimizer, use_cuda, **kwargs):
    model.train()

    stats = defaultdict(list)
    beta = kwargs['beta']
    for x in train_loader:
        if use_cuda:
            x = x.cuda()
        losses = model.loss(x)
        optimizer.zero_grad()
        loss = losses['recon_loss'] + beta * losses['kl_loss']
        loss.backward()
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
    use_kl_annealing=False, # loss should be dict contained keys 'recon_loss' and 'kl_loss'
    annealing_kwargs={
        'begin' : 0,
        'end' : 1,
        'anneal_fn': lambda x: x
    },
    visualization_samples_enabled=False, # model should have 'sample' method
    sample_kwargs={},
    loss_key='total_loss',
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    
    if use_cuda:
        model = model.cuda()

    train_params_dict = {} # passed to train_epoch function
    if use_kl_annealing:
        train_epoch_fn = train_epoch_vae_kl_anneal
        assert('begin' in annealing_kwargs)
        assert('end' in annealing_kwargs)
        assert('anneal_fn' in annealing_kwargs)
        begin = annealing_kwargs['begin']
        end = annealing_kwargs['end']
        assert(begin < end)
        anneal_fn = annealing_kwargs['anneal_fn']

        assert(hasattr(anneal_fn, '__call__'))
        beta = np.linspace(begin, end, epochs)
    else:
        train_epoch_fn = train_epoch
        train_params_dict['loss_key'] = loss_key

    if visualization_samples_enabled:
        samples_ctx = init_samples_vis_ctx()

    for epoch in forrange:
        model.train()
        
        if use_kl_annealing:
            train_params_dict['beta'] = anneal_fn(beta[epoch])

        train_loss = train_epoch_fn(model, train_loader, optimizer, use_cuda, **train_params_dict)
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
                    samples = samples.cpu()
                show_epoch_samples(samples_ctx, samples, title=f'Samples (epoch={epoch})')

    return dict(train_losses), dict(test_losses)
