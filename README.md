# Portable ES
Portable ES is a distributed gradient-less optimization framework built on PyTorch & Numpy.
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
  def reset(self) -> torch.Tensor;
  # Get stats/dist from a single run; can be found in tensorboard
  # Format {'scalars': {[str]: float}, 'images': {[str]: numpy.ndarray}}
  def eval(self) -> dict;
```

## Configuring your training
Check out `examples/train.py` for a full example with configuration.

## FAQ
### Why the *_args/*_kwargs?
These are the standard way to refer to pythonic arguments, and is done partially for optimization and partially for compatibility.
This allows you to use other enviroments (like OpenAI Gyms), and other models without the overhead of copying/transfering large memory-blocks/tensors over the network.

We might change this later on, but for now this works.

### Why can't I use the standard PyTorch optimizers?
While it is possible to use regular optimizers with custom gradients in pytorch, they might cause some overhead/flexibility issues down the line.
An additional consideration is that traditional pytorch optimizers are not as easily modifiable/readable.

We may add support for the regular PyTorch optimizers later on.

## Roadmap
- [ ] Implement and test ESAC (ES+SAC)
- [ ] Make it more modular
  - [ ] Make `reward_norm` extendable
  - [ ] Create a hyperparameter scheduler
- [ ] Further Profile portable_es
- [ ] Write-up on zeroth-order optimization
- [ ] Recurrent wrapper