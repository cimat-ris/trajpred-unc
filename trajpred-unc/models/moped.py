import torch
import torch.nn as nn
import torch.nn.functional as F

def get_rho(sigma, delta):
    """
    sigma is represented by softplus function  'sigma = log(1 + exp(rho))' to make sure it
    remains always positive and non-transformed 'rho' gets updated during backprop.
    """
    rho = torch.log(torch.expm1(delta * torch.abs(sigma)) + 1e-20)
    return rho

def MOPED_layer(layer, det_layer, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes. AAAI 2020.
    """
    if (str(layer) == 'Conv2dReparameterization()'):
        #set the priors
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize surrogate posteriors
        layer.mu_kernel.data = det_layer.weight.data
        layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif (isinstance(layer, nn.Conv2d)):
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data2

    elif (str(layer) == 'LinearReparameterization()'):
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize the surrogate posteriors

        layer.mu_weight.data = det_layer.weight.data
        layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif str(layer).startswith('Batch'):
        #initialize parameters
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data
        layer.running_mean.data = det_layer.running_mean.data
        layer.running_var.data = det_layer.running_var.data
        layer.num_batches_tracked.data = det_layer.num_batches_tracked.data
