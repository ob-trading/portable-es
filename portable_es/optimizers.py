import torch
import numpy as np

#! Note: use Zeros_like (and similar) so the device/requires_grad get transferred as well
class Optimizer:
    set_params = False # Overwrite params instead of updating

    # All memory alloc here
    def reset(self, num_params, flat_init):
        self.dim = num_params

    # Computation here
    def compute_grads(self, origin_g, stepsize):
        raise NotImplementedError

    # Per batch-entry processing here if needed
    def process_subgrad(self, g_hat):
        pass

class BasicSGD(Optimizer):
    def compute_grads(self, origin_g, stepsize):
        return stepsize * origin_g

class SGD(Optimizer):
    def __init__(self, momentum=0.9):
        self.momentum = momentum

    def reset(self, num_params, flat_init):
        super().reset(num_params, flat_init)
        self.v = torch.zeros_like(flat_init)

    def compute_grads(self, origin_g, stepsize):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * origin_g
        return stepsize * self.v

class Adam(Optimizer):
    def __init__(self, beta1=0.99, beta2=0.999, epsilon=1e-8):
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def reset(self, num_params, flat_init):
        super().reset(num_params, flat_init)
        self.m = torch.zeros_like(flat_init)
        self.v = torch.zeros_like(flat_init)
        self.step = 0

    def compute_grads(self, origin_g, stepsize):
        self.step += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * origin_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (origin_g ** 2)

        m_corr = self.m/(1-self.beta1)
        v_corr = self.v/(1-self.beta2)

        return stepsize * m_corr / (torch.sqrt(v_corr) + self.epsilon)

class AdaBelief(Optimizer):
    def __init__(self, beta1=0.99, beta2=0.999, epsilon=1e-8):
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def reset(self, num_params, flat_init):
        super().reset(num_params, flat_init)
        self.m = torch.zeros_like(flat_init)
        self.s = torch.zeros_like(flat_init)
        self.step = 0

    def compute_grads(self, origin_g, stepsize):
        self.step += 1
        self.m = (self.beta1 * self.m) + ((1 - self.beta1) * origin_g)
        self.s = (self.beta2 * self.s) + (1 - self.beta2) * ((origin_g - self.m) ** 2)

        m_corr = self.m/(1-self.beta1)
        s_corr = (self.s + self.epsilon)/(1-self.beta2)

        return (stepsize * m_corr) / (torch.sqrt(s_corr) + self.epsilon)

class AdaMM(Optimizer):
    def __init__(self, beta1=0.99, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.v_init = 1e-7 #0.00001

    def reset(self, num_params, flat_init):
        super().reset(num_params, flat_init)
        self.m = torch.zeros_like(flat_init)
        self.v_hat = self.v_init * torch.ones_like(flat_init)
        self.v = self.v_init * torch.ones_like(flat_init)

    def compute_grads(self, origin_g, stepsize):
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * origin_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (origin_g ** 2)
        self.v_hat = torch.max(self.v_hat,self.v)
        # TODO: Paper says diag(v_hat), but reference impl says raw :thinking:
        # V_hat = torch.diag(self.v_hat)
        delta = stepsize * self.m / torch.sqrt(self.v_hat)
        return delta


class AdaScale(Optimizer):
    def __init__(self, T_SI = 2000, lr_base = 0.01, lr_decay = 0.999, epsilon = 1e-6):
        self.tau = 0
        self.eps = epsilon
        self.lr_base = lr_base
        self.lr_decay = lr_decay

        self.mean_norm = None
        self.b = 0
        
    def reset(self, num_params, flat_init):
        super().reset(num_params, flat_init)

    def process_subgrad(self, g_hat):
        if self.mean_norm == None:
            self.mean_norm = torch.norm(g_hat) ** 2
        else:
            self.mean_norm += torch.norm(g_hat) ** 2
        self.b += 1
        

    def compute_grads(self, origin_g, stepsize):
        aggr_norm = torch.norm(origin_g) ** 2 #

        # (E[ 1/s * sum(||g_t^(i)||**2) ] + eps) / (E[||g_t||**2 ] + eps)
        r_t = (float(self.mean_norm / self.b) + self.eps) / (float(aggr_norm) + self.eps)
        lr = self.lr_base * self.lr_decay ** np.floor(self.tau)
        self.tau += r_t

        # Cleanup
        self.mean_norm = None
        self.b  = 0
        
        return (r_t * lr) * origin_g
    
class RAdam(Optimizer):
    def __init__(self, beta1=0.99, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.p_inf = 2/(1-self.beta2) - 1
        self.epsilon = epsilon

    def reset(self, num_params, flat_init):
        super().reset(num_params, flat_init)
        # self.v_hat = self.v_init * torch.ones((self.dim,), requires_grad=False)
        self.m = torch.zeros_like(flat_init)
        self.v = torch.zeros_like(flat_init)
        self.step = 0

    def compute_grads(self, origin_g, stepsize):
        self.step += 1
        self.v = self.beta2 * self.v + (1 - self.beta2) * (origin_g ** 2)
        self.m = self.beta1 * self.m + (1 - self.beta1) * origin_g
        mt_hat = self.m/(1-(self.beta1 ** self.step))
        _b2sq = self.beta2 ** self.step
        pt = self.p_inf - 2 * self.step * _b2sq/(1-_b2sq)
        if pt > 4:
            v_hat = torch.sqrt(self.v/(1-_b2sq))
            rt = np.sqrt(((pt-4)*(pt-2)*self.p_inf)/((self.p_inf-4)*(self.p_inf-2)*pt))
            return stepsize*rt*mt_hat/(v_hat + self.epsilon)
        else:
            return stepsize*mt_hat


class LookAhead(Optimizer):
    set_params = True # LookAhead maintains model params

    def __init__(self, base_optimizer, k=10, alpha=0.5):
        self.opt = base_optimizer
        self.k = k
        self.a = alpha
        self.step = 0

    def reset(self, num_params, flat_init):
        super().reset(num_params, flat_init)
        self.slow = flat_init.clone().detach()
        self.fast = flat_init.clone().detach()
        self.opt.reset(num_params, flat_init)

    def compute_grads(self, origin_g, stepsize):
        self.step += 1
        if self.step % self.k == 0:
            self.fast += self.opt.compute_grads(origin_g, stepsize)
            self.slow += (self.a * (self.fast - self.slow))
            self.fast.data.copy_(self.slow)
        else:
            self.fast += self.opt.compute_grads(origin_g, stepsize)

        return self.fast