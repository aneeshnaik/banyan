# banyan

Bayesian neural network in PyTorch. Theory and implementation described in [Naik & Widmark (2022)](https://arxiv.org/abs/2206.04102).

## Usage

Three objects are exposed to the user in `banyan`:
- `BLinear`: Bayesian linear layer, i.e. linear layer in which weights and biases are randomly sampled (from Gaussian distributions) on each forward pass.
- `BNN`: Network comprising series of `BLinear` layers with non-linear activation functions.
- `BLoss`: Loss function comparing ensemble of `BNN` predictions with known output labels.

The notebooks in the `examples` directory give illustrative examples of usage and training.

## Prerequisites

The only prerequisites are PyTorch (2.0.0) and NumPy (1.23.5). The version numbers indicated here are the versions used in the development and testing of `banyan`, rather than specific recommendations. Earlier/later versions are likely to work too.
