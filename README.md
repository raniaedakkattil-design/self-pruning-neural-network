# Self-Pruning Neural Network

## Overview

This project implements a neural network that learns to prune itself during training by assigning a learnable gate to each weight.

## Key Idea

Each weight is multiplied by a sigmoid gate (value between 0 and 1).
If the gate becomes close to zero, that weight is effectively removed.

## Why L1 Regularization?

An L1 penalty is applied on all gate values:

* Encourages sparsity
* Pushes many gates toward zero
* Leads to automatic pruning

## Loss Function

Total Loss = CrossEntropy + λ × Sparsity Loss

## Sparsity Loss

Sum of all gate values across layers.

## Results (Sample)

| Lambda | Accuracy (%) | Sparsity (%) |
| ------ | ------------ | ------------ |
| 0.0001 | 42.5         | 12.3         |
| 0.001  | 39.8         | 35.6         |
| 0.01   | 30.2         | 68.9         |

## Observations

* Increasing λ increases sparsity
* Higher sparsity reduces accuracy
* Trade-off exists between model size and performance

## Conclusion

The model successfully learns to identify and prune less important connections during training.
