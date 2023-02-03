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
| 2 | September, 19 | <b>Lecture:</b> Autoregressive models (WaveNet, PixelCNN). Bayesian Framework. Latent Variable Models (LVM). Variational lower bound (ELBO). | [slides](lectures/lecture2/Lecture2.pdf) |
|  |  | <b>Seminar:</b> MADE theory and practice. PixelCNN implementation hints. | [notebook](seminars/seminar2/seminar2.ipynb) |
| 3 | September, 26 | <b>Lecture:</b> EM-algorithm, amortized inference. ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). | [slides](lectures/lecture3/Lecture3.pdf) |
|  |  | <b>Seminar:</b> CNNs. Additional notes on autoregressive models. Latent Variable Models. | [notebook](seminars/seminar3/seminar3.ipynb)<br>[CNNs_note](seminars/seminar3/convolutions.pdf) |
| 4 | October, 3 | <b>Lecture:</b> VAE limitations. Posterior collapse and decoder weakening. Tighter ELBO (IWAE). Normalizing flows prerequisities. | [slides](lectures/lecture4/Lecture4.pdf) |
|  |  | <b>Seminar:</b> Gaussian Mixture Models (GMM). MLE for GMM, ELBO and EM-algorithm for GMM. VAE basics.  | [notebook](seminars/seminar4/seminar4.ipynb) |
| 5 | October, 10 | <b>Lecture:</b> Normalizing Flow (NF) intuition and definition. Forward and reverse KL divergence for NF. Linear flows. | [slides](lectures/lecture5/Lecture5.pdf) |
|  |  | <b>Seminar:</b> VAE implementation hints. IWAE model. | [notebook](seminars/seminar5/seminar5.ipynb) |
| 6 | October, 17 | <b>Lecture:</b> Autoregressive flows (gausian AR NF/inverse gaussian AR NF). Coupling layer (RealNVP). NF as VAE model. | [slides](lectures/lecture6/Lecture6.pdf) |
|  |  | <b>Seminar:</b> Flows. Planar flows. Forward KL vs Reverse KL. Planar flows via Forward KL and Reverse KL. | [notebook](seminars/seminar6/seminar6.ipynb)<br>[planar_flow_practice](seminars/seminar6/planar_flow.ipynb)<br>[autograd_jacobian](seminars/seminar6/jacobian_note.ipynb) |
| 7 | October, 24 | <b>Lecture:</b> 7. Discrete data vs continuous model. Model discretization (PixelCNN++). Data dequantization: uniform and variational (Flow++). ELBO surgery and optimal VAE prior. Flow-based VAE prior. | [slides](lectures/lecture7/Lecture7.pdf) |
|  |  | <b>Seminar:</b> RealNVP hints. Discretization of continuous distribution (MADE++). | [notebook](seminars/seminar7/seminar7.ipynb) |
| 8 | October, 31 | <b>Lecture:</b> 8. Flows-based VAE posterior vs flow-based VAE prior. Likelihood-free learning. GAN optimality theorem. | [slides](lectures/lecture8/Lecture8.pdf) |
|  |  | <b>Seminar:</b> VAE with learnable prior. Aggregated posterior. Integer discrete flows.  | [notebook](seminars/seminar8/seminar8.ipynb) |
| 9 | November, 7 | <b>Lecture:</b> Vanishing gradients and mode collapse, KL vs JS divergences. Adversarial Variational Bayes. Wasserstein distance. Wasserstein GAN (WGAN). | [slides](lectures/lecture9/Lecture9.pdf) |
|  |  | <b>Seminar:</b> KL vs JS divergences. Mode collapse. Vanilla GAN in 1D coding. | [notebook](seminars/seminar9/GAN_colab.ipynb)<br>[notebook_done](seminars/seminar9/GAN_colab_with_code.ipynb) |
| 10 | November, 14 | <b>Lecture:</b> WGAN with gradient penalty (WGAN-GP). Spectral Normalization GAN (SNGAN). f-divergence minimization. GAN evaluation. | [slides](lectures/lecture10/Lecture10.pdf) |
|  |  | <b>Seminar:</b> 1-Wasserstein distance introduction: discrete and continuous case. WGAN theory. Vanilla GAN on 2D data practice. | [notebook](seminars/seminar10/seminar10.ipynb)<br>[WGAN_theory](seminars/seminar10/Continuous_1_wasserstein_note.pdf) |
| 11 | November, 21 | <b>Lecture:</b> GAN evaluation (Inception score, FID, Precision-Recall, truncation trick). Discrete VAE latent representations. | [slides](lectures/lecture11/Lecture11.pdf) |
|  |  | <b>Seminar:</b> WGANs on multimodal 2D data. GANs zoo. Evolution of GANs. StyleGAN implementation: start discussion. | [notebook](seminars/seminar11/seminar11.ipynb)<br>[GANevolution](seminars/seminar11/GANs_evolution_and_StyleGAN.pdf)<br>[StyleGAN_not_completed](seminars/seminar11/StyleGAN.ipynb) |
| 12 | November, 28 | <b>Lecture:</b> Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). Neural ODE. Adjoint method. | [slides](lectures/lecture12/Lecture12.pdf) |
|  |  | <b>Seminar:</b> VQ-VAE implementation hints. StyleGAN coding and assessing. | [notebook](seminars/seminar12/seminar12.ipynb)<br>[StyleGAN](seminars/seminar12/StyleGAN_final.ipynb) |
| 13 | December, 5 | <b>Lecture:</b> Continuous-in-time NF (FFJORD, Hutchinson's trace estimator). Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. Score matching. | [slides](lectures/lecture13/Lecture13.pdf) |
|  |  | <b>Seminar:</b> CNF theory. Langevin Dynamics. Score matching practice. | [notebook](seminars/seminar13/seminar13.ipynb) |
| 14 | December, 12 | <b>Lecture:</b> Noise conditioned score network (NCSN). Gaussian diffusion process. Denoising diffusion probabilistic model (DDPM). | [slides](lectures/lecture14/Lecture14.pdf) |
|  |  | <b>Seminar:</b> NCSN and DDPM: theory and implementation on 2D data. | [notebook](seminars/seminar14/seminar14.ipynb) |
|  | December, 19 | <b>Oral exam</b> |  |

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | September, 22 | October, 6 | <ol><li>Theory (KDE, MADE, alpha-divergences).</li><li>PixelCNN on MNIST.</li><li>PixelCNN autocomplete and receptive field.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw1.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-AIMasters-course/blob/main/homeworks/hw1.ipynb)  |
| 2 | October, 6 | October, 20 | <ol><li>Theory (log-derivative trick, IWAE theorem, EM-algorithm for GMM).</li><li>VAE on 2D data.</li><li>VAE on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw2.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-AIMasters-course/blob/main/homeworks/hw2.ipynb) |
| 3 | October, 20 | November, 3 | <ol><li>Theory (Sylvester flows, NF Expressivity).</li><li>RealNVP on 2D data.</li><li>RealNVP on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw3.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-AIMasters-course/blob/main/homeworks/hw3.ipynb) |
| 4 | November, 3 | November, 17 | <ol><li>Theory (Mixture of Logistics, MI in ELBO surgery).</li><li>VAE with AR decoder on MNIST.</li><li>VAE with AR prior on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw4.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-AIMasters-course/blob/main/homeworks/hw4.ipynb) |
| 5 | November, 17 | December, 1 | <ol><li>Theory (IW dequantization, LSGAN, GP theorem).</li><li>WGAN/WGAN-GP/SN-GAN on CIFAR10.</li></ol>  | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw5.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-AIMasters-course/blob/main/homeworks/hw5.ipynb) |
| 6 | December, 1 | December, 15 | <ol><li>Theory (Neural ODE vs backprop, Gumbel-Max trick).</li><li>FID and Inception Score.</li><li>VQ-VAE with PixelCNN prior.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw6.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-AIMasters-course/blob/main/homeworks/hw6.ipynb) |

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
