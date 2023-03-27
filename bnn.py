#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian neural network.

Created: July 2022
Author: A. P. Naik
"""
import torch
from .blinear import BLinear


class BNN(torch.nn.Module):
    """
    Bayesian neural network, inherits torch module.

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
    act_fn : function, optional
        Activation function (from torch.nn) for network units. The default is
        ReLU.
    act_kwargs : dict, optional
        Keyword arguments to feed to act_fn. The default is no arguments (empty
        dict).

    """

    def __init__(
        self, N_in, N_out, N_hidden, N_units,
        sigma_initial=0.1, mumax_initial=1.0,
        act_fn=torch.nn.ReLU, act_kwargs={}
    ):
        """
        Create instance of BNN (see class docs for more info).

        Module list is constructed iteratively, layer by layer, then used to
        initialise a torch Sequential.
        """
        # initialise parent (nn.Module)
        super().__init__()

        # store attrs
        self.N_in = N_in
        self.N_out = N_out
        self.N_hidden = N_hidden
        self.N_units = N_units
        self.sigma_initial = sigma_initial
        self.mumax_initial = mumax_initial
        self.act_fn = act_fn
        self.act_kwargs = act_kwargs

        # multi-dim output not yet implemented.
        if N_out != 1:
            raise NotImplementedError("N_out != 1 not yet implemented.")

        # construct
        self.layers = self.__construct_network()

        return

    def __construct_network(self):
        """Construct sequence of BLinear layers and activation functions."""

        # convenience variables
        Nh = self.N_hidden
        Nu = self.N_units
        Ni = self.N_in
        No = self.N_out

        # BLinear args
        blargs = {
            'sigma_initial': self.sigma_initial,
            'mumax_initial': self.mumax_initial
        }

        # if no hidden layers, single layer in->out.
        # otherwise loop over hidden layers
        layers = torch.nn.ModuleList([])
        if Nh == 0:
            layers.append(BLinear(Ni, No, **blargs))

        else:

            # check if N_units array-like (specifying #units in each layer)
            # if so, check length of N_units equals no. of hidden layers
            # otherwise cast int N_units into [N_units, N_units, ...]
            if hasattr(Nu, "__len__"):
                lu = len(Nu)
                if lu != Nh:
                    msg = f"len(N_units)={lu} but N_hidden={Nh}. " \
                          "Should be equal."
                    raise ValueError(msg)
            else:
                Nu = [Nu for _ in range(Nh)]

            # input layer
            layers.append(BLinear(Ni, Nu[0], **blargs))
            layers.append(self.act_fn(**self.act_kwargs))

            # interior layers
            for i in range(Nh - 1):
                layers.append(BLinear(Nu[i], Nu[i + 1], **blargs))
                layers.append(self.act_fn(**self.act_kwargs))

            # output layer
            layers.append(BLinear(Nu[-1], No, **blargs))

        return layers

    def forward(self, x, N_samples):
        """
        Perform N_samples forward passes through network.

        Parameters
        ----------
        x : torch.Tensor, shape (N_batch, N_in)
            Input data.
        N_samples : int
            Number of samples to generate from network

        Returns
        -------
        y : torch.Tensor, shape (N_batch, N_samples, N_out)
            Network outputs.

        """
        # tile x so shape (N_batch, N_in) -> (N_batch, N_samples, N_in)
        y = x[:, None].tile((1, N_samples, 1))

        # iterate through layers
        for layer in self.layers:
            y = layer(y)
        return y

    def save(self, fname):
        """Save model state_dict at fname."""
        torch.save(self.state_dict(), fname)
        return

    def load(self, fname, device):
        """Load model state_dict from fname onto given torch.device."""
        state_dict = torch.load(fname, map_location=device)
        self.load_state_dict(state_dict)
        return
