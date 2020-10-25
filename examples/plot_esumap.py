from portable_es import ModelWrapper
import torch
import copy
import tqdm
import time
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

import numpy as np
import umap
import umap.plot

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# TODO: move plot_types, value_types & UMAP(...) to params
if __name__ == "__main__":
    tracedata = torch.load('checkpt-trace.pt')
    rtrace = tracedata['raw_trace'] # tracedata['trace']
    ntrace = tracedata['trace'] # tracedata['trace']
    # print(tracedata)
    # exit()

    torch.manual_seed(2)
    _model = tracedata['model_class'](*tracedata['model_args'], **tracedata['model_kwargs'])
    model = ModelWrapper(_model)
    model.eval()
    model.set_config(tracedata)

    # print(rtrace)

    params = []
    rewards = []
    plot_types = ['master',  'global']
    global_sampling = 8
    trace = list(zip(ntrace, rtrace))
    trace = trace[-500:]

    for epoch, repoch in tqdm.tqdm(trace):
        for plot_type in plot_types:
            if plot_type == 'master':
                model.model = copy.deepcopy(_model)
                # print(repoch)
                model.update_from_epoch(epoch)
                    
                params.append(parameters_to_vector(model.parameters()).detach().numpy())
                rewards.append(np.mean(list(repoch.values())) * 2)
            elif plot_type == 'global':
                for i, seed in enumerate(tqdm.tqdm(epoch['deltas'])):
                    if i % global_sampling != 0:
                        continue
                    model.model = copy.deepcopy(_model)
                    model.apply_seed(seed)
                    
                    params.append(parameters_to_vector(model.parameters()).detach().numpy())
                    rewards.append(repoch[seed])
        model.model = _model
        model.update_from_epoch(epoch)

    # mapper = umap.UMAP(n_neighbors=100, n_epochs=500).fit(params)
    mapper = umap.UMAP(n_neighbors=50, n_epochs=400, min_dist=0.005).fit(params)

    valuetypes = ['arange', 'rewards', 'cos', 'param_magnitude']

    for valuetype in tqdm.tqdm(valuetypes):
        if valuetype == 'arange':
            values = np.arange(len(params))
        elif plot_type == 'pop_arange':
            values = np.arange(len(params)) // (tracedata['popsize'] // global_sampling)
        elif valuetype == 'rewards':
            values = np.array(rewards)
        elif valuetype == 'cos':
            values = np.arange(len(params)) // (tracedata['popsize'] // global_sampling)
            values %= 3
            values += 1
            values[0] = 0
        elif valuetype == 'param_magnitude':
            values = np.array([np.mean(x) for x in params])

        plot = umap.plot.interactive(mapper, values=values, width=1800, height=1600, theme='fire')
        # plot = umap.plot.points(mapper, values=np.array(rewards), theme='fire')
        umap.plot.show(plot)
        time.sleep(2)

    # plot = umap.plot.diagnostic(mapper, diagnostic_type='pca')
    # umap.plot.show(plot)
    # plot = umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
    # umap.plot.show(plot)
    plot = umap.plot.connectivity(mapper, show_points=True)
    umap.plot.show(plot)
    
