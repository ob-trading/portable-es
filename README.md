# Portable ES
Portable ES is a distributed optimization framework built on PyTorch & Numpy.
It's main focus is autoregressive reinforcement learning scenarios.
A lot of the code from PyTorch has been replaced as they are not intended for this purpose.
It was originally made as an Evolutionary Strategies (ES) implementation.

## Features
* Dynamic distribution
  * Add/remove worker nodes over networks
  * No additional code required
  * Infinitely Distributable
  * CPU & GPU possible
* Easy setup
  * Get going in less time than a regular PyTorch setup
* Reuse PyTorch model/architectures
* Optimized for quick training
* Very simplistic implementation

## Caveats
* Heterogeneous computing isn't always deterministic
  * While this hasn't been an issue so far, it could end up degrading training performance
* Config isn't very intuitive because of performance optimization & distributed-ness

## Getting Started
## Creating an enviroment
First thing you'll need is an enviroment, ideally these are deterministic but at larger minibatches/populations this doesn't matter as much.
It uses a similar interface as OpenAI's Gym project, however it is somewhat simplified.

```python
class EnvInterface:
  def __init__(self, **kwargs); # note: no args
  # step through enviroment, maps: action -> newstate, reward, done
  def step(self, action) -> Tuple[torch.Tensor, float, bool];
  # will change the seed of the current batch; called per epoch
  def randomize(self, np_randomstate: np.random.RandomState);
  # reset internal parameters, called per episode
  def reset(self);
```

## Configuring your training

Example Config:
```python
config = {
    # Hyperparameters
    'sigma': 0.05,
    'sigma_decay': 0.99999,
    'lr': 0.01,
    'lr_decay': 0.99999,
    'optimizer': Adam(),           
    'popsize': 42,                 # Amount of model perbutations per epoch (aka population)
    'antithetic': False,           # Add the inverse of the population to get a symetric gradient evaluation
    'reward_norm': 'ranked',       # Method of reward normalization: ranked, stdmean (aka Z-score)
    'epochs': 10000,              
    'device': 'cpu',               # Device same as in pytorch; All nodes should have this device (`cuda` without postfix will randomize the cuda device)
    'pre_training_epochs': 0,      # De-randomizes enviroment for N steps (useful for testing if your model works)
    'logdir': 'lstm-adam-1',       # Tensorboard: runs/lstm-adam-1

    # Model/Env config
    'model_class': LSTMClassifier,
    'model_args': (1, 3),
    'model_kwargs': {},
    'env_eval_every': 5,
    'env_class': SimuGym,
    'env_config': {},              
    'env_episodes': 5,             # Amount of minibatches
}
```

## FAQ
### Why the *_args/*_kwargs?
These are the standard way to refer to pythonic arguments, and is done partially for optimization and partially for compatibility.
This allows you to use other enviroments (like OpenAI Gyms), and other models without the overhead of copying/transfering large memory-blocks/tensors over the network.

We might change this later on, but for now this works

### Why can't I use the standard PyTorch optimizers?
While it is possible to use regular optimizers with custom gradients in pytorch, they might cause some overhead/flexibility issues down the line.
An additional consideration is that traditional pytorch optimizers are not as easily modifiable/readable.

## Roadmap
- [ ] Implement and test ESAC (ES+SAC)
- [ ] Make it more modular
  - [ ] Make `reward_norm` extendable
  - [ ] Create a hyperparameter scheduler
- [ ] Further Profile portable_es
- [ ] Write-up on zeroth-order optimization
- [ ] Recurrent wrapper