import math
import copy
import torch

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

# {x: y[], a: b[]} -> [{x: y[0], a: b[0]}, {x: y[0], a: b[1]}, ....]
# Generates matrix of all permutations in a dictionary of arrays
# If non array is provided it is assumed to be a static value
def generate_matrix(matrix: dict):
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