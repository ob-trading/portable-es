import torch
import numpy as np
from collections import defaultdict
from .utils import params2vector, create_deco_meta


#! Note: use Zeros_like (and similar) so the device/requires_grad get transferred as well
class Optimizer:
    set_params = False # Overwrite params instead of updating
    internal_mut = False # Modifying parameters in-place (1.4+)

    def __init__(self, steps_per_epoch=0):
        self.step = 0
        self.steps_per_epoch = steps_per_epoch

    # All memory alloc here
    def reset(self, num_params, flat_init, param_shapes, mut_model):
        self.dim = num_params
        self.step = 0

    # Computation here
    def compute_grads(self, origin_g, model, stepsize):
        self.step += 1
        self.epoch = (self.step // self.steps_per_epoch) + 1

    # Per batch-entry processing here if needed
    def process_subgrad(self, g_hat):
        pass

class BasicSGD(Optimizer):
    def compute_grads(self, origin_g, model, stepsize):
        return stepsize * origin_g

class SGD(Optimizer):
    def __init__(self, momentum=0.9):
        self.momentum = momentum

    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)
        self.v = torch.zeros_like(flat_init, requires_grad=False)

    def compute_grads(self, origin_g, model, stepsize):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * origin_g
        return stepsize * self.v

class Adam(Optimizer):
    def __init__(self, beta1=0.99, beta2=0.999, epsilon=1e-8, debias=True, adamD=True, steps_per_epoch=1):
        super().__init__(steps_per_epoch=steps_per_epoch)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.adamD = adamD
        self.debias = debias
    
    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)
        self.m = torch.zeros_like(flat_init, requires_grad=False)
        self.v = torch.zeros_like(flat_init, requires_grad=False)

    def compute_grads(self, origin_g, model, stepsize):
        super().compute_grads(origin_g, model, stepsize)
        self.m = self.beta1 * self.m + (1 - self.beta1) * origin_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (origin_g ** 2)

        m_corr = self.m/(1-self.beta1)
        v_corr = self.v/(1-self.beta2)
       
        # Stepsize debiasing
        if adamD:
            alpha = stepsize * np.sqrt(1-self.beta2**self.epoch)
        elif self.debias:
            alpha = stepsize * (np.sqrt(1-self.beta2**self.epoch) / (1-self.beta1))

        return alpha * (m_corr / (torch.sqrt(v_corr) + self.epsilon))

class AdaBelief(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, steps_per_epoch=1):
        super().__init__(steps_per_epoch=steps_per_epoch)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)
        self.m = torch.zeros_like(flat_init, requires_grad=False)
        self.s = torch.zeros_like(flat_init, requires_grad=False)

    def compute_grads(self, origin_g, model, stepsize):
        super().compute_grads(origin_g, model, stepsize)
        self.m = (self.beta1 * self.m) + ((1 - self.beta1) * origin_g)
        self.s = (self.beta2 * self.s) + (1 - self.beta2) * ((origin_g - self.m) ** 2)

        m_corr = self.m/(1-self.beta1**self.epoch)
        s_corr = (self.s + self.epsilon)/(1-self.beta2**self.epoch)

        return (stepsize * m_corr) / (torch.sqrt(s_corr) + self.epsilon)

class AdaMM(Optimizer):
    def __init__(self, beta1=0.99, beta2=0.999):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.v_init = 1e-7 #0.00001

    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)
        self.m = torch.zeros_like(flat_init, requires_grad=False)
        self.v_hat = self.v_init * torch.ones_like(flat_init, requires_grad=False)
        self.v = self.v_init * torch.ones_like(flat_init, requires_grad=False)

    def compute_grads(self, origin_g, model, stepsize):
        self.m = self.beta1 * self.m + (1 - self.beta1) * origin_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (origin_g ** 2)
        self.v_hat = torch.max(self.v_hat,self.v)
        # TODO: Paper says diag(v_hat), but reference impl says raw :thinking:
        # V_hat = torch.diag(self.v_hat)
        delta = stepsize * self.m / torch.sqrt(self.v_hat)
        return delta


class AdaScale(Optimizer):
    def __init__(self, T_SI = 2000, lr_base = 0.01, lr_decay = 0.999, epsilon = 1e-6):
        super().__init__()
        self.tau = 0
        self.eps = epsilon
        self.lr_base = lr_base
        self.lr_decay = lr_decay

        self.mean_norm = None
        self.b = 0
        
    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)

    def process_subgrad(self, g_hat):
        if self.mean_norm == None:
            self.mean_norm = torch.norm(g_hat) ** 2
        else:
            self.mean_norm += torch.norm(g_hat) ** 2
        self.b += 1
        

    def compute_grads(self, origin_g, model, stepsize):
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
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.p_inf = 2/(1-self.beta2) - 1
        self.epsilon = epsilon

    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)
        # self.v_hat = self.v_init * torch.ones((self.dim,), requires_grad=False)
        self.m = torch.zeros_like(flat_init, requires_grad=False)
        self.v = torch.zeros_like(flat_init, requires_grad=False)
        self.step = 0

    def compute_grads(self, origin_g, model, stepsize):
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

class DecoupledWeightDecay(Optimizer):
    """
    As used in AdamW but in a wrapper module instead.
    
    `DecoupledWeightDecay(Adam())` will give you AdamW

    weight_decay should be approximately the same as in other frameworks unless you enable `use_stepsize`, 
    which will require a larger weight_decay as it is multiplied by the learning rate; 
    this is for usage with schedulers.
    """
    def __init__(self, base_optimizer, weight_decay=1e-5, use_stepsize=False):
        super().__init__()
        self.opt = base_optimizer
        assert not self.opt.set_params and not self.opt.internal_mut, f"Optimizer doesn't work with {type(self.opt).__name__} optimizer"
        self.wd = weight_decay
        self.use_stepsize = use_stepsize

    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)
        self.opt.reset(num_params, flat_init, param_shapes, mut_model)

    def compute_grads(self, origin_g, model, stepsize):
        delta = self.opt.compute_grads(origin_g, model, stepsize)
        cparams = params2vector(model.parameters()).detach()
        if self.use_stepsize:
            return delta - (stepsize * self.wd * cparams)
        return delta - (self.wd * cparams)

class LookAhead(Optimizer):
    set_params = True # LookAhead maintains model params

    def __init__(self, base_optimizer, k=10, alpha=0.5):
        super().__init__()
        self.opt = base_optimizer
        assert not self.opt.set_params and not self.opt.internal_mut, f"Optimizer doesn't work with {type(self.opt).__name__} optimizer"
        self.k = k
        self.a = alpha
        self.step = 0

    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)
        self.slow = flat_init.clone().detach()
        self.fast = flat_init.clone().detach()
        self.opt.reset(num_params, flat_init, param_shapes, mut_model)

    def compute_grads(self, origin_g, model, stepsize):
        self.step += 1
        if self.step % self.k == 0:
            self.fast += self.opt.compute_grads(origin_g, model, stepsize)
            self.slow += (self.a * (self.fast - self.slow))
            self.fast.data.copy_(self.slow)
        else:
            self.fast += self.opt.compute_grads(origin_g, model, stepsize)

        return self.fast

def neuron_norm(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.norm(dim=1).view(*view_shape)
    else:
        return x.abs()

def neuron_mean(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.mean(dim=1).view(*view_shape)
    else:
        raise Exception("neuron_mean not defined on 1D tensors.")

class PytorchAdapter(Optimizer):
    internal_mut = True
    
    def __init__(self, optimizer_cls, *args, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.args = args
        self.kwargs = kwargs

    def reset(self, num_params: int, flat_init, param_shapes, mut_model):
        self.optimizer = self.optimizer_cls(mut_model.parameters(),*self.args, **self.kwargs)

    def compute_grads(self, origin_g, model, stepsize):
        idx = 0
        for param in model.parameters():
            size = np.product(param.data.shape)
            block_g = -(origin_g[idx:idx+size].view(param.data.shape))
            param.grad = block_g

            idx += size

        self.optimizer.step()


class Nero(Optimizer):
    internal_mut = True

    def __init__(self, lr=0.01, beta=0.999, constraints=True, steps_per_epoch=1):
        super().__init__(steps_per_epoch=steps_per_epoch)
        self.lr = lr
        self.beta = beta
        self.constraints = constraints
        self.steps_per_epoch = steps_per_epoch

    def reset(self, num_params: int, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)

        self.state = defaultdict(dict)
        for i, param in enumerate(mut_model.parameters()):
            if self.constraints and param.dim() > 1:
                param -= neuron_mean(param)
                param /= neuron_norm(param)

            self.state[i]['exp_avg_sq'] = torch.zeros_like(neuron_norm(param))
            self.state[i]['scale'] = neuron_norm(param).mean()
            if self.state[i]['scale'] == 0.0:
                self.state[i]['scale'] == 0.01


    def compute_grads(self, origin_g, model, stepsize):
        super().compute_grads(origin_g, model, stepsize)
        idx = 0
        for i, param in enumerate(model.parameters()):
            size = np.product(param.data.shape)
            block_g = -(origin_g[idx:idx+size].view(param.data.shape))

            bias_correction = 1 - self.beta ** self.epoch
            self.state[i]['exp_avg_sq'] = self.beta * self.state[i]['exp_avg_sq'] + (1 - self.beta) * neuron_norm(block_g)**2
            g_normed = block_g / (self.state[i]['exp_avg_sq']/bias_correction).sqrt()
            g_normed[torch.isnan(g_normed)] = 0

            param.data -= stepsize * self.state[i]['scale'] * g_normed

            if self.constraints and param.dim() > 1:
                param.data -= neuron_mean(param)
                param.data /= neuron_norm(param)

            idx += size


class NovoGrad(Optimizer):
    internal_mut = True
    # b1=[0.9,0.95], b2=[0.2,0.5]
    def __init__(self, beta1=0.9, beta2=0.5, weight_decay=0.002, epsilon=1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.d = weight_decay
        self.eps = epsilon

    def reset(self, num_params, flat_init, param_shapes, mut_model):
        super().reset(num_params, flat_init, param_shapes, mut_model)

        self.m = []
        for s in param_shapes:
            self.m.append(torch.zeros(s, dtype=flat_init.dtype, device=flat_init.device, requires_grad=False))
        self.v = np.zeros(len(param_shapes))
        self.step = 0

    def compute_grads(self, origin_g, model, stepsize):
        self.step += 1
        idx = 0
        i = 0
        for param in model.parameters():
            size = np.product(param.data.shape)
            block = origin_g[idx:idx+size].view(param.data.shape)

            if self.step == 1:
                # Init v & m
                self.v[i] = torch.norm(block) ** 2
                self.m[i] = block/np.sqrt(self.v[i]) + self.d * param.data
            else:
                self.v[i] = self.beta2 * self.v[i] + (1-self.beta1) * torch.norm(block) ** 2
                self.m[i] = self.beta1 * self.m[i] + (block/(np.sqrt(self.v[i]) + self.eps) + self.d * param.data)
                param.data += self.m[i] * stepsize
            
            i += 1
            idx += size
        
