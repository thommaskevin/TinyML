# layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with Gaussian variational posterior.
    Each weight and bias has a mean and log-variance parameter.
    """
    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters: mean near zero, logvar small negative."""
        std = math.sqrt(1. / self.in_features)
        self.weight_mu.data.normal_(0, std)
        self.weight_logvar.data.fill_(-6)   # small variance
        self.bias_mu.data.zero_()
        self.bias_logvar.data.fill_(-6)

    def forward(self, x, sample=True):
        """
        Forward pass. If sample=True, draw weights from posterior;
        otherwise use the mean (deterministic).
        """
        if sample:
            weight = self.reparameterize(self.weight_mu, self.weight_logvar)
            bias = self.reparameterize(self.bias_mu, self.bias_logvar)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self):
        """
        Compute KL divergence between the variational posterior
        and a standard normal prior N(0, prior_var).
        """
        sigma = torch.exp(0.5 * self.weight_logvar)
        kl = 0.5 * torch.sum(
            -1 - self.weight_logvar + (self.weight_mu ** 2 + sigma ** 2) / self.prior_var
        )
        if self.bias_mu is not None:
            sigma_b = torch.exp(0.5 * self.bias_logvar)
            kl += 0.5 * torch.sum(
                -1 - self.bias_logvar + (self.bias_mu ** 2 + sigma_b ** 2) / self.prior_var
            )
        return kl


class BayesianConv2d(nn.Module):
    """
    Bayesian 2D convolutional layer.
    (Simplified version; extends similarly to BayesianLinear.)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_var=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.prior_var = prior_var

        # Variational parameters for weights: (out_channels, in_channels, kH, kW)
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        )
        self.weight_logvar = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        )
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        std = math.sqrt(1. / n)
        self.weight_mu.data.normal_(0, std)
        self.weight_logvar.data.fill_(-6)
        self.bias_mu.data.zero_()
        self.bias_logvar.data.fill_(-6)

    def forward(self, x, sample=True):
        if sample:
            weight = self.reparameterize(self.weight_mu, self.weight_logvar)
            bias = self.reparameterize(self.bias_mu, self.bias_logvar)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self):
        sigma = torch.exp(0.5 * self.weight_logvar)
        kl = 0.5 * torch.sum(
            -1 - self.weight_logvar + (self.weight_mu ** 2 + sigma ** 2) / self.prior_var
        )
        if self.bias_mu is not None:
            sigma_b = torch.exp(0.5 * self.bias_logvar)
            kl += 0.5 * torch.sum(
                -1 - self.bias_logvar + (self.bias_mu ** 2 + sigma_b ** 2) / self.prior_var
            )
        return kl