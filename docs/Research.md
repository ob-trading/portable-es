**This document contains unique or yet to be implemeted improvements to portable-es**
Note some of these are theoritical and untested

# Block-wise update annealing for ES
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

# CartpoleV1 SpeedRun
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

