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