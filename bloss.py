#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic kernel density loss criterion.

Created: July 2022
Author: A. P. Naik
"""
import torch
import numpy as np


class BLoss(torch.nn.Module):
    r"""
    Creates a logistic kernel density loss criterion, comparing predictions
    model(x) with labels y.

    NOTE: CURRENTLY ONLY IMPLEMENTED FOR 1D OUTPUT.

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

    Parameters (applies to generated loss function)
    -----------------------------------------------
    preds : torch Tensor, (N_batch, N_samples, N_out) or (N_samples, N_out)
        Predictions for input data.
    labels : torch Tensor, (N_batch, N_out) or (N_out)
        Labels for input data.

    Returns (applies to generated loss function)
    --------------------------------------------
    loss : torch Tensor, 0-dimensional
        Logistic kernel density loss.

    """

    def __init__(self):
        """Create BLoss function (see class docs for more info)."""
        super(BLoss, self).__init__()
        return

    def forward(self, preds, labels):
        """Loss criterion forward pass."""
        # if only single (batch size 1), unsqueeze
        if preds.ndim == 2:
            preds = preds[None]
            labels = labels[None]

        # infer number of samples
        N_samples = preds.shape[1]

        # kernel bandwidth
        h = 0.6 * torch.std(preds, dim=1) * np.power(N_samples, -0.2)
        h = h[:, None]

        # evaluate kernel pdf
        z = 0.5 * (labels[:, None] - preds) / h
        sech2 = 1 / (4 * h * torch.cosh(z)**2)
        kernel = torch.sum(sech2, axis=1) / N_samples
        lnkernel = torch.log(kernel)

        # loss
        loss = -torch.sum(lnkernel) / len(labels)

        return loss
