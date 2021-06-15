import math
import copy
import torch
import numpy as np
import types
import inspect
from typing import Any, Callable, List, TypeVar
from functools import wraps

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def generate_matrix(matrix: dict):
    """
    Generates matrix of all permutations in a dictionary of arrays
    If non array is provided it is assumed to be a static value

    >>> {x: y[], a: b[]} -> [{x: y[0], a: b[0]}, {x: y[0], a: b[1]}, ....]
    """
    x = list(matrix)[0]
    m = copy.deepcopy(matrix) # TODO: fix non-picklable iterables (i.e. generators)
    del m[x]
    
    # Statoc values
    if type(matrix[x]) != list:
        matrix[x] = [matrix[x]]

    for y in matrix[x]:
        if m:
            for om in generate_matrix(m):
                yield {x: y, **om}
        else:
            yield {x: y}

def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


# TODO: generalize to apply_vec2param and apply_param2vec?
def params2vector(params, shapes=None):
    if not shapes:
        shapes = []
        for param in params:
            shapes.append(param.data.shape)

    NPARAMS = np.sum([np.prod(x) for x in shapes])
    vec = torch.zeros((NPARAMS,), requires_grad=False)
    idx = 0
    i = 0
    for param in params:
        size = np.product(shapes[i])
        vec[idx:idx+size] = param.data.view(-1)
        idx += size
        i += 1

    return vec

FuncT = TypeVar("FuncT", bound=Callable[..., Any])

def create_deco_meta(wrappers: List[FuncT]):
    class DecoMeta(type):
        def __new__(cls, name, bases, attrs):
            for attr_name, attr_value in attrs.items():
                if isinstance(attr_value, types.FunctionType):
                    attrs[attr_name] = cls.deco(attr_value)

            return super().__new__(cls, name, bases, attrs)

        @classmethod
        def deco(cls, func: FuncT) -> FuncT:
            prev = func
            for wraps in reversed(wrappers):
                #print(f'wrapping {prev.__name__} with {wraps.__name__}')
                prev = wraps(prev)
            return prev
    
    return DecoMeta

