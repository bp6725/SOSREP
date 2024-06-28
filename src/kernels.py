#This file contains the implementation of the kernels used in the SOSREP paper in torch.
# This is important given the scikit-learn implementation is not compatible with the torch tensors.
# Namely, the kernels are:
# 1. The SDO kernel
# 2. The RBF set kernel
# 3. The Laplacian kernel

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from functools import partial

def th2np(tensor):
    return tensor.detach().cpu().numpy()

class RBfKernel():
    """
    Compute the RBF kernel between two tensors.

    :param X1: Tensor of shape (N, D)
    :param X2: Tensor of shape (N, D)
    :param gamma: Scale parameter
    :return: Kernel matrix of shape (N, N)
    """

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X1, X2):
        dim = X1.shape[1]
        distance_squared = torch.cdist(X1.unsqueeze(0), X2.unsqueeze(0), p=2).squeeze(0) ** 2
        kernel_matrix = (torch.exp(-(0.5) * (self.gamma ** 2) * distance_squared/dim) +
                         torch.min(torch.exp(-(0.5) * (self.gamma ** 2) * distance_squared/dim)).clone().detach().item() * 1e-6)

        return kernel_matrix

class LaplacianKernel():
    """
        Compute the Laplacian kernel between two tensors.

        :param X1: Tensor of shape (N, D)
        :param X2: Tensor of shape (M, D)
        :param gamma: Scale parameter for the Laplacian kernel
        :return: Kernel matrix of shape (N, M)
        """
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X1, X2):
        dim = X1.shape[1]

        distance_squared = torch.cdist(X1.unsqueeze(0), X2.unsqueeze(0), p=2).squeeze(0)

        kernel_matrix = torch.exp(-1 * self.gamma * distance_squared / np.sqrt(dim))
        kernel_matrix = kernel_matrix + torch.min(kernel_matrix).clone().detach().item() * 1e-6
        return kernel_matrix

class SDoKernel():
    def __init__(self, b_coef, data_dim, n_samples_r_theta = 10000,rnd_seed=None, tau = 1,
                 grid_upper_bound = 500, grid_size = 30000, device = "cuda"):
        self.data_dim = data_dim
        self.als = np.zeros(data_dim // 2 + 1)
        self.als[-1] = 1.
        self.als = self.als * tau

        self.tau = tau

        self.n_pder = len(self.als)
        self.n_samples_r_theta = n_samples_r_theta

        self.b_coef = b_coef

        self.grid_upper_bound = grid_upper_bound
        self.grid_size = grid_size

        self.Z = None
        self.bt = None

        self.rnd_seed = rnd_seed
        self.rng = np.random.default_rng(rnd_seed)
        self.device = device

        self.initialize_samples()

    def initialize_samples(self):
        self.Z = self.sample_z_on_vz2() if self.Z is None else self.Z

        if self.bt is None:
            self.bt = torch.Tensor(
                self.rng.random(
                    size=(1, self.Z.shape[1])
                ) * 2 * np.pi
            )

    def __call__(
            self,
            X1,
            X2
    ):
        self.initialize_samples()
        Z = self.Z
        bt = self.bt.to(self.device)

        # We assume X:[n_points, d] ; Z:[d, n_samples]. So : X*Z -> rows are per point. columns are over samples
        XZ = X1.matmul(Z)
        YZ = X2.matmul(Z)

        XZb_cos = torch.cos(XZ + bt)
        YZb_cos = torch.cos(YZ + bt)

        # K: [n_samples,n_samples]
        K = (1 / (Z.shape[1])) * XZb_cos.matmul(YZb_cos.T)

        return K

    def sample_z_on_vz2(self):
        r_samples = self.sample_wr_grid()

        theta_samples = self.sample_theta(self.data_dim, self.n_samples_r_theta, rng=self.rng)

        # r_samples_th = torch.concatenate(r_samples)
        r_samples_th = torch.concat(r_samples).to(self.device)

        theta_samples_th = torch.Tensor(theta_samples[0:len(r_samples)]).to(self.device)

        z = (r_samples_th * theta_samples_th.T)

        self.r_samples = r_samples

        return z

    def sample_theta(self, d, n_samples_theta, rng):
        _samples = rng.normal(size=(n_samples_theta, d))

        _norms = np.linalg.norm(_samples, axis=1) ** (-1)
        return _samples * _norms[:, np.newaxis]

    def find_inverse(self, wr_log_cdf, log_limit):
        low, high = 0.1, 1000
        last, mid = 0.1, high / 2
        while abs(mid - last) > 1e-12:
            if wr_log_cdf(mid) < log_limit:
                low = mid
            else:
                high = mid
            last, mid = mid, (low + high) / 2
        return mid

    def simple_wr_func(self, als, b_coef, d, r):
        if r <= 0: return torch.Tensor([-1 * 10 ** (6)])

        r = th2np(r)[0]

        # m = len(als) + 1
        m = len(als)

        _x = (b_coef * r) ** 2

        log_denominator = logsumexp(np.array([0, m * np.log(_x)]))

        res = -log_denominator + (d - 1) * np.log(r)

        return torch.Tensor([res])

    def sample_wr_grid(self):
        ''' For automatic r_limit
        def sample_wr_grid(wr_eval, als, b_coef, d, n_samples_r, get_f_vals = False, pdf_weight_range=0.999, grid_size = 10 ** (5)):
         _wr_cdf_log_func = partial(wr_log_analytical_cdf, d, d // 2 + 1, b_coef)
        _wr_log_func = partial(wr_eval, als, b_coef, d)

        grid_upper_bound = find_upper_bound(_wr_cdf_log_func, pdf_weight_range)
        grid = np.linspace(1e-8, grid_upper_bound,grid_size)

        '''
        _wr_log_func = partial(self.simple_wr_func, self.als, self.b_coef, self.data_dim)

        grid = np.linspace(1e-5, self.grid_upper_bound, self.grid_size)

        f_vals = np.array(
            [th2np(_wr_log_func(torch.Tensor([x])))[0] for x in grid]
        )

        log_sum = logsumexp(f_vals)
        f_vals -= log_sum
        f_vals = np.exp(f_vals)
        f_vals /= f_vals.sum()

        samples = self.rng.choice(grid, p=f_vals, size=self.n_samples_r_theta)

        samples = [torch.Tensor([x]) for x in samples]

        return samples

