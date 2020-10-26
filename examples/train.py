import os
import gym
import torch
import numpy as np
from typing import Tuple

def cartpole_cast(action):
    return int(action.argmax().cpu().detach())

class GymWrapper:
    def __init__(self, gym_name='CartPole-v1', castf=lambda x: x, **kwargs):
        self.g = gym_name
        self.castf = castf

    def step(self, action) -> Tuple[torch.Tensor, float, bool]:
        obs, reward, done, info = self.env.step(self.castf(action))
        return torch.from_numpy(obs).float(), reward, done

    def randomize(self, np_randomstate: np.random.RandomState):
        pass

    def reset(self):
        if not getattr(self, 'env', None):
            self.env = gym.make(self.g)
        return torch.from_numpy(self.env.reset()).float()

    def eval(self):
        return {}


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
from portable_es.optimizers import Adam

config = {
    # Hyperparameters
    'sigma': 0.1,
    'sigma_decay': 0.9999,
    'lr': 0.03,         # Big population so we can afford a high learning rate
    'lr_decay': 0.99999,
    'optimizer': Adam(),
    'popsize': 256,
    'antithetic': True,
    'reward_norm': 'ranked',  # ranked, stdmean # TODO: topk
    'epochs': 1000,
    'device': 'cpu',
    'pre_training_epochs': 0,
    'logdir': 'cartpole1-rnn-1',

    'model_class': RNNClassifier,
    'model_args': (2,),
    'model_kwargs': {'channels': 4, 'hidden': 5, 'layers': 2},

    'env_eval_every': 5,
    'env_class': GymWrapper,
    'env_config': {'gym_name': 'CartPole-v1', 'castf': cartpole_cast},
    'env_episodes': 1,
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