# Optimizers
## CartPole-v1
config (omitted unrelated/redundant fields):
```
{
    'sigma': 0.1,
    'sigma_decay': 0.9999,
    'lr': 0.03,
    'lr_decay': 0.99999,
    'popsize': 256,
    'antithetic': True,
    'reward_norm': 'ranked',

    'model_class': RNNClassifier,
    'model_args': (2,),
    'model_kwargs': {'channels': 4, 'hidden': 5, 'layers': 2},

    'env_class': GymWrapper,
    'env_config': {'gym_name': 'CartPole-v1', 'castf': argmax_cast},
    'env_episodes': 1,
}
```

### SpeedRun
25 Epochs using the model from `train.py` in `examples`.
```
1. Novograd v1  (Nov 2020; `<hash> `; 497.6,  σ=18.76)
1. AdaBelief    (Oct 2020; `9a4a3ac`; 484.1,  σ=41.00)
2. Adam v2      (Oct 2020; `9a4a3ac`; 220.9,  σ=82.99)
3. AdaMM v2     (Oct 2020; `e6776de`; 129.0, σ=112.14)
4. Adam v1      (Sep 2020; `19c49a0`;  97.6,  σ=61.92)
5. AdaMM v1     (Sep 2020; `19c49a0`;  68.4,  σ=39.49)
6. Radam v1     (Oct 2020; `b0dd8e1`;  48.6,  σ=52.29)
```


**Notes:**
* AdaMM was used with Adam's default parameters instead of the recommended from paper
  * Likely because they are targeting Zeroth-Order opt. they kept the smoothing low
* Radam includes a type of warmup, which might explain the low score


## Acrobot-v1
config (omitted unrelated/redundant fields):
```
config = {
    # Hyperparameters
    'sigma': 0.1,
    'sigma_decay': 0.9999,
    'lr': 0.01,
    'lr_decay': 0.99999,
    'popsize': 256,
    'antithetic': True,
    'reward_norm': 'ranked',

    'model_class': RNNClassifier,
    'model_args': (3,),
    'model_kwargs': {'channels': 6, 'hidden': 5, 'layers': 3},

    'env_class': GymWrapper,
    'env_config': {'gym_name': 'Acrobot-v1', 'castf': argmax_cast},
    'env_episodes': 1,
}
```

### SpeedRun
25 Epochs using the model from `train.py` in `examples`.
```
1. AdaBelief    (Oct 2020; `295cf5d`; -82.1,  σ=26.26)
2. Adam v2      (Oct 2020; `295cf5d`; -84.9,  σ=34.51)
5. Novograd v1  (Nov 2020; `<hash> `; -86.3,  σ=42.03)
3. AdaMM v2     (Oct 2020; `295cf5d`; -102.0, σ=73.68)
4. RAdam v1     (Oct 2020; `295cf5d`; -211.6, σ=182.32)
```