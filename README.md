# Deep Generative Models course, AIMasters, 2022-2023

## Description
The course is devoted to modern generative models (mostly in the application to computer vision). 

We will study the following types of generative models: 
- autoregressive models, 
- latent variable models, 
- normalization flow models, 
- adversarial models,
- diffusion models.

Special attention is paid to the properties of various classes of generative models, their interrelationships, theoretical prerequisites and methods of quality assessment.

The aim of the course is to introduce the student to widely used advanced methods of deep learning.

The course is accompanied by practical tasks that allow you to understand the principles of the considered models.

## Materials

| # | Date | Description | Slides |
|---------|------|-------------|---------|
| 0 | September, 12 | Logistics and intro. | [slides](lectures/intro.pdf) |
| 1 | September, 12 | <b>Lecture:</b> Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive modelling. | [slides](lectures/lecture1/Lecture1.pdf) |
|  |  | <b>Seminar:</b> Introduction. Maximum likelihood estimation. Kernel density estimation. Histograms and KDE. | [notebook](seminars/seminar1/seminar1.ipynb) |
| 2 | September, 19 | <b>Lecture:</b> Autoregressive models (WaveNet, PixelCNN). Bayesian Framework. Latent Variable Models. Variational lower bound (ELBO). | [slides](lectures/lecture2/Lecture2.pdf) |
|  |  | <b>Seminar:</b> MADE theory and practice. PixelCNN implementation hints. | [notebook](seminars/seminar2/seminar2.ipynb) |
| 3 | September, 26 | <b>Lecture:</b> EM-algorithm, amortized inference. ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). | TBA |
|  |  | <b>Seminar:</b> Mean field approximation. | TBA |
| 4 | October, 3 | <b>Lecture:</b> Posterior collapse and decoder weakening. Tighter ELBO (IWAE). Idea of normalizing flows. | TBA |
|  |  | <b>Seminar:</b> EM-algorithm. VAE theory. Automatic differentiation through random graph. | TBA |
| 5 | October, 10 | <b>Lecture:</b> Flow models definition. Forward and reverse KL divergence. Linear flows (Glow). Residual flows (Planar/Sylvester flows). | TBA |
|  |  | <b>Seminar:</b> IWAE theory. IWAE variational posterior. VAE vs Normalizing flows. | TBA |
| 6 | October, 17 | <b>Lecture:</b> Autoregressive flows (MAF/IAF). Coupling layer (RealNVP). | TBA |
|  |  | <b>Seminar:</b> Planar flows. Forward vs Reverse KL. | TBA |
| 7 | October, 24 | <b>Lecture:</b> Uniform and variational dequantization. ELBO surgery and optimal VAE prior. Flows-based VAE posterior vs flow-based VAE prior. | TBA |
|  |  | <b>Seminar:</b> VAE prior (VampPrior). SurVAE. RealNVP hints. | TBA |
| 8 | October, 31 | <b>Lecture:</b> Disentanglement learning (beta-VAE, DIP-VAE + summary). Likelihood-free learning. GAN theorem. | TBA |
|  |  | <b>Seminar:</b> GAN vs VAE vs NF. GAN in 1d coding. | TBA |
| 9 | November, 7 | <b>Lecture:</b> Vanishing gradients and mode collapse, KL vs JSD. Adversarial Variational Bayes. Wasserstein distance. | TBA |
|  |  | <b>Seminar:</b> GAN vs VAE theory. KL vs JS divergences. | TBA |
| 10 | November, 14 | <b>Lecture:</b> Wasserstein GAN. WGAN-GP. Spectral Normalization GAN. f-divergence minimization. | TBA |
|  |  | <b>Seminar:</b> WGAN: practice. Optimal transport task. SN-GAN: practice. | TBA |
| 11 | November, 21 | <b>Lecture:</b> GAN evaluation (Inception score, FID, Precision-Recall, truncation trick). GAN models (Self-Attention GAN, BigGAN, PGGAN, StyleGAN). | TBA |
|  |  | <b>Seminar:</b> StyleGAN: implementation hints. | TBA |
| 12 | November, 28 | <b>Lecture:</b> 12. Discrete VAE latent representations. Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). Neural ODE. | TBA |
|  |  | <b>Seminar:</b> NeuralODE explanation. | TBA |
| 13 | December, 5 | <b>Lecture:</b> Adjoint method. Continuous-in-time NF (FFJORD, Hutchinson's trace estimator). Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. | TBA |
|  |  | <b>Seminar:</b> TBA | TBA |
| 14 | December, 12 | <b>Lecture:</b> Score matching. Noise conditioned score network (NCSN). Denoising diffusion probabilistic model (DDPM). | TBA |
|  |  | <b>Seminar:</b> TBA | TBA |
|  | December, 19 | <b>Oral exam</b> | TBA |

## Homeworks 
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | September, 22 | October, 6 | <ol><li>Theory (MADE, Histogram).</li><li>PixelCNN on MNIST.</li><li>PixelCNN autocomplete and receptive field.</li></ol> |  |
| 2 | October, 6 | October, 20 | <ol><li>Theory (log-derivative trick, IWAE theorem).</li><li>VAE on 2D data.</li><li>VAE on CIFAR10.</li></ol> | TBA |
| 3 | October, 20 | November, 3 | <ol><li>Theory (Sylvester flows).</li><li>RealNVP on 2D data.</li><li>RealNVP on CIFAR10.</li></ol> | TBA |
| 4 | November, 3 | November, 17 | <ol><li>Theory (MI in ELBO surgery).</li><li>VAE with AR decoder on MNIST.</li><li>VAE with AR prior on CIFAR10.</li></ol> | TBA |
| 5 | November, 17 | December, 1 | <ol><li>Theory (IW dequantization, LSGAN).</li><li>WGAN/WGAN-GP on 2D data.</li><li>WGAN-GP on CIFAR10.</li></ol> | TBA |
| 6 | December, 1 | December, 14 | <ol><li>Theory (Neural ODE backprop).</li><li>SN-GAN on CIFAR10.</li><li>FID and Inception Score.</li></ol> | TBA |

## Game rules
- 6 homeworks each of 13 points = **78 points**
- oral cozy exam = **26 points**
- maximum points: 78 + 26 = **104 points**
### Final grade: `floor(relu(#points/8 - 2))`

## Previous episodes
- [2022, spring, OzonMasters](https://github.com/r-isachenko/2022-DGM-Ozon-course)
- [2021, autumn, MIPT](https://github.com/r-isachenko/2021-DGM-MIPT-course)
- [2021, spring, OzonMasters](https://github.com/r-isachenko/2021-DGM-Ozon-course)
- [2020, autumn, MIPT](https://github.com/r-isachenko/2020-DGM-MIPT-course)

## Author, feel free to contact :)

- **telegram:** [@roman_isachenko](https://t.me/roman_isachenko)
- **e-mail:** roman.isachenko@phystech.edu
