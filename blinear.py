#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian linear layer.

Created: March 2022
Author: A. P. Naik
"""
import numpy as np
import torch


class BLinear(torch.nn.Module):
    """
    Bayesian linear layer, inherits torch.nn.Module.

    Parameters
    ----------
    N_in : int
        Number of inputs.
    N_out : int
        Number of outputs.
    sigma_initial : float, optional
        Initial distribution widths for all weights and biases. The default is
        0.1.
    mumax_initial : float, optional
        Initial distribution means are sampled uniformly between -mumax_initial
        and mumax_initial. The default is 1.0.

    """

    def __init__(self, N_in, N_out, sigma_initial=0.1, mumax_initial=1.0):
        """Create instance of BLinear layer (see class docs for more info)."""
        # initialise nn.Module
        super().__init__()

        # save attrs
        self.N_in = N_in
        self.N_out = N_out

        # set up weight (w) and bias (b) distributions
        self.w_mu = torch.nn.Parameter(torch.Tensor(N_out, N_in))
        self.w_lsigma = torch.nn.Parameter(torch.Tensor(N_out, N_in))
        self.b_mu = torch.nn.Parameter(torch.Tensor(N_out))
        self.b_lsigma = torch.nn.Parameter(torch.Tensor(N_out))

        # initialise values
        self.w_mu.data.uniform_(-mumax_initial, mumax_initial)
        self.w_lsigma.data.fill_(np.log(sigma_initial))
        self.b_mu.data.uniform_(-mumax_initial, mumax_initial)
        self.b_lsigma.data.fill_(np.log(sigma_initial))

        return

    def forward(self, x, N_samples):
        """Forward pass through layer."""
        # sample weights and biases
        dev = self.w_mu.device
        r1 = torch.randn((N_samples, self.N_out, self.N_in), device=dev)
        r2 = torch.randn((N_samples, self.N_out), device=dev)
        w = self.w_mu + torch.exp(self.w_lsigma) * r1
        b = self.b_mu + torch.exp(self.b_lsigma) * r2

        # y = Wx + b
        y = torch.matmul(x, torch.transpose(w, -1, -2)) + b[:, None]
        return y

    def extra_repr(self):
        """Extra information provided for print(BLinear) call."""
        return f'N_in={self.N_in}, N_out={self.N_out}'