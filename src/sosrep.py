import torch
import numpy as np
from scipy.signal import argrelextrema
from functools import partial
import copy
import random
from tqdm import tqdm

from SOSREP.src.kernels import RBfKernel, SDoKernel, LaplacianKernel, LaplacianLogKernel, RBFLogKernel

def np2th(np_arr, device='cuda'):
    if isinstance(np_arr, torch.Tensor):
        if np_arr.device.type == device and np_arr.dtype == torch.float32:
            return np_arr
        else:
            return np_arr.to(device=device, dtype=torch.float32)
    else:
        return torch.tensor(np_arr, dtype=torch.float32, device=device)

def th2np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return tensor.detach().cpu().numpy()

def init_alphas(Ndata, device, rng_seed, requires_grad=True):
    rng = np.random.default_rng(rng_seed)

    rnd_arr = rng.normal(size=(Ndata, 1))
    rnd_arr = np.abs(rnd_arr)

    init = np2th(rnd_arr, device)
    alphas = torch.autograd.Variable(
        .1 * init,
        requires_grad=requires_grad)

    return alphas

def fit_mds_kernel(X_train, kernel_func, learning_rate_scalar, n_iters_optim, rng_seed,
                   grad_natural=False, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = np2th(X_train, device)
    Ndata, input_dim = x_train.shape

    K = kernel_func(x_train, x_train)

    alphas = init_alphas(Ndata, device, rng_seed, requires_grad=False)

    learning_rate = learning_rate_scalar

    res_loss = []
    res_norm = []
    res_mean_ll = []
    res_grad_norm_lst = []
    res_grad_Hnorm_lst = []

    with torch.no_grad():
        with tqdm(total=n_iters_optim) as p:
            for cur_iter in range(n_iters_optim):
                Ka = K @ alphas

                direct_grad = 2 * (alphas - (1. / Ka) / Ndata)

                sq_norm = torch.reshape((alphas.T) @ Ka, ())
                lls = torch.log(Ka ** 2)
                mean_ll = lls.mean()

                loss = sq_norm - mean_ll

                alphas.add_(direct_grad, alpha= -1 * learning_rate)

                res_mean_ll.append(th2np(mean_ll))
                res_norm.append(th2np(sq_norm))
                res_loss.append(th2np(loss))

                grad_norm = (direct_grad ** 2).sum()
                grad_Hnorm = torch.reshape(direct_grad.T @ (K @ direct_grad), ())

                res_grad_norm_lst.append(th2np(grad_norm).reshape(()))
                res_grad_Hnorm_lst.append(th2np(grad_Hnorm).reshape(()))

                p.update(1)
                if verbose and cur_iter % 2000 == 0:
                    print(f'sq_norm: {sq_norm:.5f}, mean_ll: {mean_ll:.5f}, loss: {loss:.5f}')

    return alphas, K, res_loss, res_grad_norm_lst, res_grad_Hnorm_lst

def evaluate_f2(X_train, X_test, kernel_func, alphas):
    x_train = np2th(X_train)
    e_train = kernel_func(x_train, x_train)
    f_xtrain = torch.matmul(e_train, alphas)

    if X_test is not None:
        x_test = np2th(X_test)
        e_test = kernel_func(x_test, x_train)
        f_xtest = torch.matmul(e_test, alphas)
        return f_xtest**2, f_xtrain**2
    else:
        return f_xtrain**2

class SOSREP:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, X_test=None,stable_kernels = True, kernel_type='RBF', Bs=np.logspace(-2, 2, 15),
            div_window=3, learning_rate_scalar=1e-2, n_iters_optim=30000, rng_seed=19372):
        self.X_train = X_train = np2th(X_train, self.device)
        X_test = np2th(X_test, self.device) if X_test is not None else None

        if stable_kernels :
            kernel_type = 'Log' + kernel_type if 'Log' not in kernel_type else kernel_type

        if X_test is None:
            # If X_test is not provided, use a portion of X_train as X_test
            train_size = int(0.8 * len(X_train))
            X_train, X_test = X_train[:train_size], X_train[train_size:]

        optimal_b, best_results, fds, Bs = self.fit_predict(
            X_train, X_test, kernel_type, Bs, div_window, learning_rate_scalar, n_iters_optim, rng_seed,stable_kernels
        )

        self.optimal_b = optimal_b
        self.kernel_func = self._return_kernel_func(kernel_type, optimal_b)
        _, _, self.alphas = best_results

        return self

    def predict(self, X):
        if self.kernel_func is None or self.alphas is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")

        f2_test, _ = evaluate_f2(np2th(self.X_train), np2th(X), self.kernel_func, self.alphas)
        return f2_test

    def fit_predict(self, X_train, X_test, kernel_type = 'RBF', Bs = np.logspace(-2,2,15),
            div_window=3, learning_rate_scalar=1e-2, n_iters_optim=30000,  rng_seed=19372, stable_kernels = True):
        def run_with_fd(b):
            kernel_func = self._return_kernel_func(kernel_type, b)
            alphas, _, _, _, _ = fit_mds_kernel(X_train, kernel_func, learning_rate_scalar, n_iters_optim,
                                                  rng_seed)
            f_xtest, f_xtrain = evaluate_f2(X_train, X_test, kernel_func, alphas)

            if 'Log' in kernel_type:
                partial_log_density = partial(self._log_density_log_kernel, X_train, kernel_func, alphas)
            else:
                partial_log_density = partial(self._log_density, X_train, kernel_func, alphas)

            fd_test = self.fast_fisher_divergence(partial_log_density, X_test)

            return fd_test, (f_xtest, f_xtrain, alphas)
        self.X_train = X_train
        if stable_kernels :
            kernel_type = 'Log' + kernel_type if 'Log' not in kernel_type else kernel_type

        X_train = np2th(X_train, self.device)
        X_test = np2th(X_test, self.device)

        fds = []
        results = []
        for b in Bs:
            fd, result = run_with_fd(b)
            fds.append(fd.detach().cpu())
            results.append(result)

        # Find stable minima
        fis_div_vls = np.log((1 + np.array(fds) + float(abs(min(fds)))))
        local_minis = argrelextrema(np.array(np.round(fis_div_vls)), np.less, order=div_window)
        optimal_index = local_minis[0][-1] if len(local_minis[0]) > 0 else -1

        if optimal_index != -1:
            optimal_b = Bs[optimal_index]
            best_results = results[optimal_index]
        else:
            # If no local minima found, choose the b with the lowest Fisher divergence
            optimal_index = np.argmin(fds)
            optimal_b = Bs[optimal_index]
            best_results = results[optimal_index]

        self.kernel_func = self._return_kernel_func(kernel_type, optimal_b)
        _, _, self.alphas = best_results


        return optimal_b, best_results, fds, Bs

    def _return_kernel_func(self, kernel_type, b):
        if kernel_type == 'SDO':
            return SDoKernel(b, self.X_train.shape[1], device=self.device)
        elif kernel_type == 'RBF':
            gamma = 1 / b
            return RBfKernel(gamma)
        elif kernel_type == 'Laplacian':
            gamma = 1 / b
            return LaplacianKernel(gamma)
        elif kernel_type == 'LogLaplacian':
            gamma = 1 / b
            return LaplacianLogKernel(gamma)
        elif kernel_type == 'LogRBF':
            gamma = 1 / b
            return RBFLogKernel(gamma)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def _calc_optimal_B_fd_online(self, f, Bs, div_window, history=None, bs_tracker=None):
        def is_bigger(f, point, fut_point, prev_history, bs_tracker):
            if prev_history[fut_point][point] == 1: return True
            if prev_history[fut_point][point] == -1: return False

            if fut_point not in bs_tracker:
                f_fu_point = f(fut_point)
                bs_tracker[fut_point] = f_fu_point
            else:
                f_fu_point = bs_tracker[fut_point]

            if point not in bs_tracker:
                f_point = f(point)
                bs_tracker[point] = f_point
            else:
                f_point = bs_tracker[point]

            _bigger = f_fu_point[0] > f_point[0]

            history[fut_point][point] = 1 if _bigger else -1
            history[point][fut_point] = -1 if _bigger else 1
            return _bigger

        if len(Bs) == 1:
            return Bs[0], f(Bs[0])[1]

        if bs_tracker is None:
            bs_tracker = {}

        if history is None:
            history = {k: {kk: 0 for kk in Bs} for k in Bs}

        if div_window == 0:
            rb = np.random.choice(Bs)
            return rb, bs_tracker[rb]

        for b in reversed(range(len(Bs))[div_window:-div_window]):
            flag = True
            for d in range(1, div_window):
                if is_bigger(f, Bs[b], Bs[b + d], history, bs_tracker) and is_bigger(f, Bs[b], Bs[b - d], history,
                                                                                     bs_tracker):
                    continue
                else:
                    flag = False
                    break

            if flag:
                return Bs[b], bs_tracker[Bs[b]][1]

        return self._calc_optimal_B_fd_online(f, Bs, div_window - 1, history, bs_tracker)

    def _log_density(self, x_train, kernel_func, alphas, x_batch):
        e = kernel_func(x_batch, x_train)
        f_vals = torch.matmul(e, alphas)
        f2_vals = f_vals ** 2
        f2_vals += torch.min(f2_vals[f2_vals != 0]) * 1e-6 if any(f2_vals != 0) else 1e-30
        return torch.log(f2_vals)

    def _log_density_log_kernel(self, x_train_as_th, log_kernel_func, alphas, x_batch):

        K_log = log_kernel_func(x_batch, x_train_as_th)
        K_log_max = torch.max(K_log, dim=1)[0][:, None]
        K_log_diff = K_log - K_log_max
        K_diff = torch.exp(K_log_diff)

        f_vals = torch.matmul(K_diff, alphas)
        f2_vals = f_vals ** 2

        return torch.log(f2_vals) + 2 * K_log_max

    def get_grads(self, f, x):
        x = x.clone().detach().requires_grad_(True)
        vals = torch.sum(f(x))
        vals.backward()
        return x.grad

    def fast_hessian_trace_estimator(self, f, x_batch, h=1e-4, eps_reps=300):
        grads = self.get_grads(f, x_batch)

        estimates = []

        for _ in range(eps_reps):
            eps = (2 * torch.randint(2, size=x_batch.shape, dtype=torch.float32) - 1).to(self.device)
            grads_shift = self.get_grads(f, x_batch + h * eps)

            trace = ((grads_shift - grads) * eps).sum(dim=1, keepdim=True) / h
            estimates.append(trace.clone().detach())

        return torch.concat(estimates, dim=1).mean(dim=1)

    def fast_fisher_divergence(self, log_density_func, x_batch, get_values=False, **args):
        grads = self.get_grads(log_density_func, x_batch)

        traces = self.fast_hessian_trace_estimator(log_density_func, x_batch, **args)

        div_per_sample = traces + 0.5 * (grads ** 2).sum(dim=1)

        if get_values:
            return div_per_sample

        return div_per_sample.mean()