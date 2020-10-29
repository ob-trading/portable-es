# Optimizers
## CartPole-v1
config (omitted unrelated/redundant fields):
```
{
    'sigma': 0.1,
    'sigma_decay': 0.9999,
    # Big population so we can afford a relatively high learning rate
    'lr': 0.03,
    'lr_decay': 0.99999,
    'popsize': 256,
    'antithetic': True,
    'reward_norm': 'ranked',

    'model_class': RNNClassifier,
    'model_args': (2,),
    'model_kwargs': {'channels': 4, 'hidden': 5, 'layers': 2},

    'env_class': GymWrapper,
    'env_config': {'gym_name': 'CartPole-v1', 'castf': cartpole_cast},
    'env_episodes': 1,
}
```

### SpeedRun
25 Epochs using the model from `train.py` in `examples`.
```
1. AdaBelief  (Oct 2020; `9a4a3ac`; 484.1,  σ=41.00)
2. Adam v2    (Oct 2020; `9a4a3ac`; 220.9,  σ=82.99)
3. AdaMM v2   (Oct 2020; `e6776de`; 129.0, σ=112.14)
4. Adam v1    (Sep 2020; `19c49a0`;  97.6,  σ=61.92)
5. AdaMM v1   (Sep 2020; `19c49a0`;  68.4,  σ=39.49)
6. Radam v1   (Oct 2020; `b0dd8e1`;  48.6,  σ=52.29)
```


**Notes:**
* AdaMM was used with Adam's default parameters instead of the recommended from paper
  * Likely because they are targeting Zeroth-Order opt. they kept the smoothing low
* Radam includes a type of warmup, which might explain the low score

