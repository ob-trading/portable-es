**This document contains unique or yet to be implemeted improvements to portable-es**
Note some of these are theoritical and untested

# Block-wise update annealing for ES
[Note: this is similar to Novograd, but using variance instead of moment]
Block here is defined as a set of homogenous parameters (e.g. Bias-for-Layer-N or Weights-for-Layer-N)

`P=Population; \pi=Policy-Parameters`
```
0. Use regular ES until variance is non-zero for 5 consecutive iterations
1. Arr <- [std(R)] * parameter-blocks
2. \theta <- Copy/Set K random parameter blocks to 1
3. For P:
  3.1. Sample \psi <- N ~ (0, 1) * \theta
  3.2. R_i <- Evaluate Policy \pi + (\psi * \sigma)
6. For block in \theta:
  6.1. Arr_block <- Arr_block * \beta_1 + std(R) * (1-\beta_1)
  6.2 \pi <- \pi + block * \sigma * mean(Arr) / Arr_block
```

# xNES for neural networks
[Discontinued]

The main issue with xNES is that it has exponential memory requirements (`torch.eye(ndim)`), this means it's pretty much infeasible to train large neural networks with it.

```
def w_k(λ, r):
    w_hp = np.log(λ/2 + 1/2)
    return (np.ones_like(r) * w_hp) - compute_ranks(r)

def mu_c(λ, n)
    mu = (5 + λ)/(5*n**1.5)
    assert mu <= 1
    return mu
```

# Diskcache for random tensors
We've tried speeding up the RNG using diskcache so it doesn't regenerate recent tensors (this theoretically should also work cross-process), however the epoch time isn't impacted much in our tests and shows a slight slow-down. We verified that it was using the caches. This indicates that the time it takes to store/load tensors is similar to the time it takes generating them on the fly.


```
from diskcache import Cache
gcache = Cache('.cache')
@gcache.memoize(expire=5 * 60, tag='radn_t')
def get_random_tensor(shape, device, seed):
  ....
```