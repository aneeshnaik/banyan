#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np
import torch
from .blinear_full import BLinearFull


class BNNFull(torch.nn.Module):
    """
    Bayesian neural network, inherits torch module.

    Parameters
    ----------
    N_in : int
        Number of inputs.
    N_out : int
        Number of outputs. NOTE: ONLY 1D OUTPUT (N_out=1) CURRENTLY
        IMPLEMENTED.
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
            layers.append(BLinearFull(Ni, No, **blargs))

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
            layers.append(BLinearFull(Ni, Nu[0], **blargs))
            layers.append(self.act_fn(**self.act_kwargs))

            # interior layers
            for i in range(Nh - 1):
                layers.append(BLinearFull(Nu[i], Nu[i + 1], **blargs))
                layers.append(self.act_fn(**self.act_kwargs))

            # output layer
            layers.append(BLinearFull(Nu[-1], No, **blargs))

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
        y : torch.Tensor, shape (N_samples, N_batch, N_out)
            Network outputs.

        """
        # tile x so shape (N_batch, N_in) -> (N_batch, N_samples, N_in)
        y = x[:, None].tile((1, N_samples, 1))

        # iterate through layers
        for layer in self.layers:
            y = layer(y)
        return y

    def calc_loss(self, x, y, N_samples):
        r"""
        Logistic kernel density loss, comparing model(x) with labels y.

        NOTE: CURRENTLY ONLY IMPLEMENTED FOR 1D OUTPUT, I.E. self.N_out = 1.

        Given input data :math:`x` (batch size :math:`N`) with *true* outputs
        :math:`\hat{y}`, the loss is calculated by generating many
        (:math:`N_\mathrm{samples}`) predictions :math:`y_{i, j}` for each data
        point :math:`i`. These predictions are then smoothed into a
        probability distribution via logistic kernel density estimation. The
        loss :math:`\mathcal{L}` is then the negative log-probability of the
        true outputs given this probability distribution:

        .. math::
            \mathcal{L} = -\frac{1}{N} \sum_i^N \ln \left[
            \frac{1}{N_\mathrm{samples}} \sum_j^{N_\mathrm{samples}}
            \frac{1}{4h_i} \mathrm{sech}^2
            \left(\frac{\hat{y}_i - y_{i, j}}{2 h_i}\right)\right],

        where the index :math:`i` is summing over the datapoints in the batch
        and index :math:`j` is summing over the samples for each data point.
        :math:`h_i` is the kernel bandwidth for data point :math:`i`.
        This is given by

        .. math::
            h_i = 0.6 \sigma_i N^{-1/5},

        where :math:`\sigma` is the standard deviation of the predictions
        :math:`y_{i, j}` for data point :math:`i`.

        Parameters
        ----------
        x : torch Tensor, shape (self.N_in) or (N_batch, self.N_in)
            Input data.
        y : torch Tensor, shape (self.N_out) or (N_batch, self.N_out)
            Labels for input data.
        N_samples : int, optional
            Number of samples to network samples to take (see maths above).

        Returns
        -------
        loss : torch Tensor, 0-dimensional
            Logistic kernel density loss.

        """

        # generate predictions
        y_pred = self(x, N_samples)

        # kernel bandwidth
        h = 0.6 * torch.std(y_pred, dim=1) * np.power(N_samples, -0.2)
        h = h[:, None]

        # evaluate kernel pdf
        sech2 = 1 / (4 * h * torch.cosh(0.5 * (y[:, None] - y_pred) / h)**2)
        kernel = torch.sum(sech2, axis=1) / N_samples
        lnkernel = torch.log(kernel)

        # loss
        loss = -torch.sum(lnkernel) / len(y)

        return loss

    def save(self, fname):
        """Save model state_dict at fname."""
        torch.save(self.state_dict(), fname)
        return

    def load(self, fname, device):
        """Load model state_dict from fname onto given torch.device."""
        state_dict = torch.load(fname, map_location=device)
        self.load_state_dict(state_dict)
        return
