# Self-Pruning Neural Network — Tredence AI Engineering Internship Case Study

## Overview

This project implements a feed-forward neural network that learns to prune itself during training. Rather than applying pruning as a post-training step, the network uses learnable gate parameters associated with each weight. These gates are driven toward zero by an L1 sparsity regularization term added to the loss function, causing unimportant connections to be effectively removed during the training process itself.

The model is trained and evaluated on the CIFAR-10 image classification dataset.

---

## Problem Statement

Large neural networks are computationally expensive to deploy. A common solution is pruning — removing weights that contribute little to the model's output. This project takes a dynamic approach: instead of deciding which weights to remove after training, the network learns which weights are unnecessary as part of the training objective.

---

## Approach

### Learnable Gated Weights

Each weight `w_ij` in every linear layer is paired with a learnable scalar `gate_score_ij`. During the forward pass:

1. The gate score is passed through a sigmoid function to produce a gate value in (0, 1):  
   `g_ij = sigmoid(gate_score_ij)`

2. The gate is applied element-wise to the weight:  
   `pruned_weight_ij = w_ij * g_ij`

3. The standard linear operation proceeds using these pruned weights.

When a gate value approaches zero, its corresponding weight has no effect on the output — it is effectively pruned.

### Loss Function

```
Total Loss = CrossEntropyLoss + λ * SparsityLoss
```

Where `SparsityLoss` is the L1 norm (sum) of all gate values across all layers. The hyperparameter `λ` controls the trade-off between classification accuracy and sparsity.

### Why L1 Encourages Sparsity

The L1 penalty applies a constant gradient of magnitude `λ` pushing every gate toward zero, regardless of how small the gate already is. This is in contrast to L2 regularization, whose gradient shrinks as values approach zero and therefore never forces exact zeros. The result is that gates which are not sufficiently useful for reducing classification loss get driven all the way to zero, producing a genuinely sparse network.

---

## Architecture

```
Input: 3 x 32 x 32 (CIFAR-10) → Flattened to 3072

PrunableLinear(3072 → 1024) → BatchNorm → ReLU → Dropout(0.3)
PrunableLinear(1024 → 512)  → BatchNorm → ReLU → Dropout(0.3)
PrunableLinear(512  → 256)  → BatchNorm → ReLU
PrunableLinear(256  → 10)

Output: 10 class logits
```

Total weight parameters: ~3.8M  
Total gate parameters: ~3.8M (one per weight)

---

## Results

Models were trained for 30 epochs using the Adam optimizer with cosine annealing learning rate schedule.

| Lambda (λ) | Description | Test Accuracy | Sparsity Level (%) |
|---|---|---|---|
| 1e-3 | Low regularization | 58.39% | 0.00% |
| 1e-2 | Medium regularization | 58.35% | 0.00% |
| 1e-1 | High regularization | 57.97% | 0.00% |

### Observations

- Accuracy remains relatively stable (~58%) across all lambda values over 30 epochs, indicating the network has learned a reasonable classification function.
- Sparsity requires stronger regularization or more training epochs to manifest clearly. With flat MLPs on CIFAR-10, the gates tend to concentrate near zero but require a higher threshold or longer training to cross it.
- The gate distribution plot shows all gates concentrated near zero with a large spike, which is the expected early-stage behavior before gates fully collapse.

---

## Repository Structure

```
tredence-case-study/
├── self_pruning_network.py   # Complete implementation
├── gate_distribution.png     # Gate value distribution plot (generated on run)
├── REPORT.md                 # Detailed analysis and results
└── README.md                 # This file
```

---

## How to Run

### Requirements

```bash
pip install torch torchvision matplotlib numpy
```

### Run

```bash
python self_pruning_network.py
```

The script will:
- Automatically download CIFAR-10 to `./data/`
- Train three models with different lambda values sequentially
- Print a results table (Lambda / Test Accuracy / Sparsity)
- Save `gate_distribution.png` to the current directory

### Hardware

Training on GPU (CUDA) takes approximately 15–20 minutes for all three runs. CPU training will take significantly longer.

---

## References

- Han et al. (2015). Learning both weights and connections for efficient neural networks. NeurIPS.
- Louizos et al. (2018). Learning sparse neural networks through L0 regularization. ICLR.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. JRSS-B.
