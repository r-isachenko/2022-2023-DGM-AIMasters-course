import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torch.distributions as TD
from PIL import Image
import itertools

def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

##############
# Taken from the seminar3

class BananaDistribution:

    def __init__(self, inv=False, device='cpu'):
        self.inv = inv
        self.x2_distrib = TD.Normal(
            torch.tensor(0.0).to(device), 
            torch.tensor(3.3).to(device))
        self.x1_distrib = TD.Normal(
            torch.tensor(0.0).to(device),
            torch.tensor(1.).to(device))

    def sample(self, count):
        if not isinstance(count, tuple):
            count = (count,)
        x2 = self.x2_distrib.sample(count)
        x1 = self.x1_distrib.sample(count) + (x2**2)/8.
        samples = torch.stack([x1, x2], axis=-1)
        if self.inv:
            samples = samples.flip(-1)
        return samples
    
    def log_prob(self, samples):
        if self.inv:
            samples = samples.flip(-1)
        x2 = samples[..., 1]
        x1 = samples[..., 0]
        log_prob_x2 = self.x2_distrib.log_prob(x2)
        log_prob_x1 = self.x1_distrib.log_prob(x1 - (x2**2)/8.)
        return log_prob_x2 + log_prob_x1
    
    def prob(self, samples):
        return torch.exp(self.log_prob(samples))

class MaskedLinear(torch.nn.Linear):
    
    
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
         
        return torch.nn.functional.linear(input, self.mask * self.weight, self.bias)


class MADE(torch.nn.Module):
    
    def __init__(self, dim, hidden_sizes, out_bins, in_bins=1):
        '''
        :Parameters:
        dim : int : number of input dimensions
        hidden_sizes : list : sizes of hidden layers
        out_bins : output params per each output dimension
        in_bins : input params per each input dimension (for example, one hot)
        '''
        super().__init__()
        
        self.dim = dim
        self.nin = dim * in_bins
        self.in_bins = in_bins
        self.nout = dim * out_bins
        self.out_bins = out_bins

        self.hidden_sizes = hidden_sizes
        # we will use the trivial ordering of input units
        self.ordering = np.arange(self.dim)

        self.net = []
        hs = [self.nin, ] + self.hidden_sizes + [self.nout, ]
        for h0, h1 in zip(hs[:-2], hs[1:-1]):
            self.net.extend([
                MaskedLinear(h0, h1),
                torch.nn.ReLU(),
            ])

        self.net.append(MaskedLinear(hs[-2], hs[-1]))
        self.net = torch.nn.Sequential(*self.net)

        self.create_mask()  # builds the initial self.m connectivity


    def create_mask(self):
        
        # 1) The ordering of input units from 1 to d (self.ordering).
        # 2) Assign the random number k from 1 to d − 1 to each hidden unit. 
        #    This number gives the maximum number of input units to which the unit can be connected.
        # 3) Each hidden unit with number k is connected with the previous layer units 
        #   which has the number less or equal than k.
        # 4) Each output unit with number k is connected with the previous layer units 
        #    which has the number less than k.

        self.assigned_numbers = {}
        self.masks = []
        L = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.assigned_numbers[-1] = self.ordering
        for l in range(L):
            self.assigned_numbers[l] = np.random.randint(
                self.assigned_numbers[l - 1].min(), self.dim - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.assigned_numbers[l - 1][:, None] <= self.assigned_numbers[l][None, :] for l in range(L)]
        masks.append(self.assigned_numbers[L - 1][:, None] < self.assigned_numbers[-1][None, :])

        masks[-1] = np.repeat(masks[-1], self.out_bins, axis=1)
        masks[0] = np.repeat(masks[0], self.in_bins, axis=0)
        self.masks = masks 

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)


    def visualize_masks(self):
        prod = self.masks[0]
        for idx, m in enumerate(self.masks):
            plt.figure(figsize=(3, 3))
            plt.title(f'layer: {idx}')
            plt.imshow(m.T, vmin=0, vmax=1, cmap='gray')
            plt.show()

            if idx > 0:
                prod=prod.dot(m)

        plt.figure(figsize=(3, 3))
        plt.title('prod')
        plt.imshow(prod.T, vmin=0, vmax=1, cmap='gray')
        plt.show()


    def forward(self, x):
        """
        :Parameters:
        x: torch.Size([BS, nin]) : input sample
        :Output:
        out : torch.Size([BS, nout]) : output 
        """
        assert len(x.size()) == 2
        assert x.shape[1] == self.nin
        batch_size = x.shape[0]
        logits = self.net(x)
        return logits

def generate_2d_image_data(count, bins=64):
    # Загружаем картинку, сжимаем к размеру (bins x bins),
    # конвертируем к grayscale - формату
    im = Image.open('pics/2d_distribution.png').resize((bins, bins)).convert('L')
    im = np.array(im).astype('float32')
    # Сейчас im : np.array размера (64, 64), 
    # элементы этого массива выглядят так:
    # 
    # array([[12., 12., 13., ...,  6.,  6.,  4.],
    #        [11., 13., 15., ...,  7.,  6.,  6.],
    #        [14., 16., 18., ...,  7.,  7.,  6.],
    #        ...,
    #        [24., 21., 25., ..., 31., 31., 24.],
    #        [18., 21., 21., ..., 26., 26., 23.],
    #        [17., 18., 20., ..., 28., 21., 19.]], dtype=float32)
    #
    # "0." - чёрный; "255." - белый

    # Здесь мы получаем двумерное категориальное распределение, 
    # с числом параметров 64 * 64
    # КОТОРОЕ МЫ И ХОТИМ ПРИБЛИЗИТЬ НАШЕЙ МОДЕЛЬЮ
    dist = im / im.sum()

    
    ### СЕМПЛИРОВАНИЕ ИЗ dist
    # pairs перечисляет все возможные пиксели
    # pairs = [(0, 0), (0, 1), ... (63, 62), (63, 63)]
    pairs = list(itertools.product(range(bins), range(bins)))
    # выбираем count пикселей в соответствии с вероятностями из dist
    idxs = np.random.choice(len(pairs), size=count, replace=True, p=dist.reshape(-1))
    samples = np.array([pairs[i] for i in idxs])

    split = int(0.8 * len(samples))
    return dist, samples[:split], samples[split:]

def plot_2d_image_data(train_data, test_data, bins):
    train_dist, test_dist = np.zeros((bins, bins)), np.zeros((bins, bins))
    for i in range(len(train_data)):
        train_dist[train_data[i][0], train_data[i][1]] += 1
    train_dist /= train_dist.sum()

    for i in range(len(test_data)):
        test_dist[test_data[i][0], test_data[i][1]] += 1
    test_dist /= test_dist.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.set_title('Train Data')
    ax1.imshow(train_dist, cmap='gray')
    ax1.axis('off')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x0')

    ax2.set_title('Test Data')
    ax2.imshow(test_dist, cmap='gray')
    ax2.axis('off')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x0')

    plt.show()