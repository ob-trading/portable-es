import os
import gym
import torch
import numpy as np
from typing import Tuple

def argmax_cast(action):
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
from portable_es.optimizers import Adam, AdaBelief, RAdam, AdaMM, NovoGrad
from portable_es.compat import GymWrapper
from portable_es.scheduler import DecayScheduler

config = {
    # Big population so we can afford a high learning rate
    'scheduler': DecayScheduler(ilr=0.03, lr_decay=0.99999, isigma=0.1, sigma_decay=0.9999),
    'optimizer': NovoGrad(),
    'popsize': 256,
    'antithetic': True,
    'reward_norm': 'ranked',  # ranked, stdmean
    'epochs': 25,
    'device': 'cpu',
    'pre_training_epochs': 0,
    'logdir': 'arcobot1-rnn-1',

    'model_class': RNNClassifier,
    'model_args': (3,),
    'model_kwargs': {'channels': 6, 'hidden': 5, 'layers': 3},

    'env_class': GymWrapper,
    'env_config': {'gym_name': 'Acrobot-v1', 'castf': argmax_cast},
    'env_episodes': 1,
    'env_eval_every': 5,
}

def params_count(model):
    return sum(p.numel() for p in model.parameters())

manager = ESManager(config)
print(f'model params: {params_count(manager.model.model)} ({params_count(manager.model.model) * 4} bytes)')

for n in range(6):
    manager.create_local_worker(ESWorker)

# For adding remote workers
print('client creds:', manager.get_client_args())

# Setup exit handler
import atexit

def exit_handler():
    torch.save(manager.model.model, 'es-model.pt')
    # For use with plot_esumap.py
    torch.save({'trace': manager.update_history,
                'raw_trace': manager.raw_history, **config}, 'checkpt-trace.pt')

atexit.register(exit_handler)

while not manager.done:
    manager.run_once()
    # Can do other (short) tasks here as well

torch.save(manager.model.model, 'es-model.pt')

# Stop all workers
manager.stop()