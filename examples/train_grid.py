import os
import gym
import torch
import numpy as np
from typing import Tuple

# An example using grid-based hyper-parameter search

def cartpole_cast(action):
    return int(action.argmax().cpu().detach())


class RNNClassifier(torch.nn.Module):
    def __init__(self, outputs, channels=1, hidden=25, layers=1, device=None):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.hidden = hidden
        self.base = torch.nn.GRU(channels, hidden, layers)

        self.classifier = torch.nn.Linear(self.hidden * self.layers, outputs)
        self.device = device
        self.reset()

    def forward(self, x):
        x = x.view((-1, 1, self.channels))
        _, self.hn = self.base(x, self.hn)

        hn = self.hn.view((-1,))
        hn = torch.sigmoid(self.classifier(hn))
        return hn

    # Handle recurrence inside the module
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        if hasattr(self, 'hn') and self.hn != None:
            self.hn = self.hn.to(self.device)

        return self

    def reset(self):
        self.base.flatten_parameters() # For CUDA
        if not hasattr(self, 'hn') or self.hn == None:
            self.hn = torch.zeros((self.layers, 1, self.hidden), device=self.device)

        self.hn[:] = 0

from portable_es import ESManager, ESWorker
from portable_es.optimizers import Adam, AdaBelief, RAdam, AdaMM
from portable_es.utils import generate_matrix
from portable_es.compat import GymWrapper

model_grid = {
    'channels': 4,
    'hidden': [5, 10, 20],
    'layers': [1, 2, 3]
}

model_matrix = generate_matrix(model_grid)

config_grid = {
    # Hyperparameters
    'sigma': 0.1,
    'sigma_decay': 0.9999,
    'lr': 0.03,         # Big population so we can afford a high learning rate
    'lr_decay': 0.99999,
    'optimizer': AdaBelief(),
    'popsize': 256,
    'antithetic': True,
    'reward_norm': 'ranked',
    'epochs': 200,
    'device': 'cpu',
    'pre_training_epochs': 0,
    'logdir': 'cartpole1-rnn-1',

    'model_class': RNNClassifier,
    'model_args': (2,),
    'model_kwargs': list(model_matrix),

    'env_eval_every': 5,
    'env_class': GymWrapper,
    'env_config': {'gym_name': 'CartPole-v1', 'castf': cartpole_cast},
    'env_episodes': 1,
}

def params_count(model):
    return sum(p.numel() for p in model.parameters())

for config in generate_matrix(config_grid):
    # TODO: Clean up declarations
    file = f'RNN{config["model_kwargs"]["hidden"]}-{config["model_kwargs"]["layers"]}.pt'
    if os.path.isfile(file):
        print(f'Skipping {file}, already trained')
        continue
    trace_file = f'RNN{config["model_kwargs"]["hidden"]}-{config["model_kwargs"]["layers"]}-trace.pt'

    config['logdir'] = f'rnn{config["model_kwargs"]["hidden"]}-{config["model_kwargs"]["layers"]}-D3-matrix'
    print(f'Running... ({file}, {config["logdir"]})')

    manager = ESManager(config)
    print('Params:', params_count(manager.model.model))

    for n in range(5):
        manager.create_local_worker(ESWorker)

    # For adding remote workers
    print('client creds:', manager.get_client_args())

    while not manager.done:
        manager.run_once()

    torch.save(manager.model.model, file)
    torch.save({'trace': manager.update_history,
                'raw_trace': manager.raw_history, **config}, trace_file)

    # Stop all workers
    manager.stop()

