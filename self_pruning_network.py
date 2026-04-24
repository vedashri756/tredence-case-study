"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Intern – Case Study

Author: [Your Name]
Description:
    Implements a feed-forward neural network with learnable gated weights
    that prune themselves during training via L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Part 1: PrunableLinear Layer

class PrunableLinear(nn.Module):
    """
    A custom Linear layer augmented with learnable gate parameters.

    Each weight w_ij has a corresponding gate score g_ij.
    The gate is computed as sigmoid(g_ij), producing a value in (0, 1).
    The effective weight used in the forward pass is: w_ij * sigmoid(g_ij).

    When a gate approaches 0, the corresponding weight is effectively pruned.
    Gradients flow through both `weight` and `gate_scores` via autograd.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias – same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores – same shape as weight
        # Initialized to small positive values so sigmoid starts near 0.5
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        # Kaiming uniform for weights (standard for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # Gate scores initialized near 0 → sigmoid ≈ 0.5 (half-open gates)
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert gate_scores → gates in (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)  # shape: (out, in)

        # Step 2: Element-wise multiply weights with gates
        pruned_weights = self.weight * gates       # shape: (out, in)

        # Step 3: Standard linear operation with pruned weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the current gate values (detached, for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# Network Definition

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32×3 = 3072 input features, 10 classes).
    All linear layers use PrunableLinear so the whole network can self-prune.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten: (B, 3, 32, 32) → (B, 3072)
        return self.net(x)

    def get_all_prunable_layers(self):
        """Returns all PrunableLinear layers in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

# Part 2: Sparsity Loss

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of all gate values across all PrunableLinear layers.

    L1 encourages sparsity because its subgradient is constant (±1),
    meaning it applies a steady 'push' toward zero regardless of magnitude —
    unlike L2, which weakens as values approach zero and never forces exact zeros.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.get_all_prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.abs().sum()  # gates ≥ 0, so abs() is a no-op here
    return total


# Part 3: Data Loading

def get_cifar10_loaders(batch_size: int = 128):
    """Returns train and test DataLoaders for CIFAR-10."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# Training Loop

def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Total Loss = Cross-Entropy + λ * SparsityLoss
        cls_loss = F.cross_entropy(logits, labels)
        sp_loss = sparsity_loss(model)
        loss = cls_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total


def compute_sparsity(model, threshold=1e-2):
    """
    Returns the fraction (%) of weights whose gate value < threshold.
    """
    all_gates = []
    for layer in model.get_all_prunable_layers():
        gates = layer.get_gates().cpu().numpy().flatten()
        all_gates.append(gates)
    all_gates = np.concatenate(all_gates)
    pruned = (all_gates < threshold).sum()
    return 100.0 * pruned / len(all_gates), all_gates


# Main Experiment

def run_experiment(lam: float, epochs: int, device, train_loader, test_loader):
    """Trains a SelfPruningNet with a given lambda and returns results."""
    print(f"\n{'='*60}")
    print(f"  Running experiment: λ = {lam}")
    print(f"{'='*60}")

    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, lam, device
        )
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            test_acc = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.1f}% | "
                  f"Test Acc: {test_acc:.1f}%")

    final_test_acc = evaluate(model, test_loader, device)
    sparsity_pct, all_gates = compute_sparsity(model)
    print(f"\n  λ={lam} → Test Acc: {final_test_acc:.2f}%  |  "
          f"Sparsity: {sparsity_pct:.2f}%")

    return {
        "lambda": lam,
        "test_accuracy": final_test_acc,
        "sparsity": sparsity_pct,
        "gates": all_gates,
        "model": model,
    }


def plot_gate_distribution(results, best_lambda):
    """Plots gate value distribution for the best model."""
    best = next(r for r in results if r["lambda"] == best_lambda)
    gates = best["gates"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full histogram
    axes[0].hist(gates, bins=100, color="#2563EB", edgecolor="white", alpha=0.85)
    axes[0].set_title(
        f"Gate Value Distribution (λ={best_lambda})\n"
        f"Sparsity: {best['sparsity']:.1f}%  |  "
        f"Test Acc: {best['test_accuracy']:.1f}%",
        fontsize=13
    )
    axes[0].set_xlabel("Gate Value (sigmoid output)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(x=0.01, color="red", linestyle="--", label="Pruning threshold (0.01)")
    axes[0].legend()

    # Zoomed view near 0 to show spike
    axes[1].hist(gates[gates < 0.1], bins=50,
                 color="#16A34A", edgecolor="white", alpha=0.85)
    axes[1].set_title("Zoom: Gates < 0.1 (spike at zero shows pruned weights)",
                      fontsize=13)
    axes[1].set_xlabel("Gate Value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to gate_distribution.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EPOCHS = 30
    LAMBDAS = [1e-3, 1e-2, 1e-1]   # Low, Medium, High — strong enough to prune

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    results = []
    for lam in LAMBDAS:
        result = run_experiment(lam, EPOCHS, device, train_loader, test_loader)
        results.append(result)

    # ── Summary Table 
    print("\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print("="*55)
    for r in results:
        print(f"{r['lambda']:<12} {r['test_accuracy']:>14.2f}% {r['sparsity']:>14.2f}%")
    print("="*55)

    # ── Best model = highest accuracy 
    best = max(results, key=lambda r: r["test_accuracy"])
    print(f"\nBest model: λ={best['lambda']} | "
          f"Acc={best['test_accuracy']:.2f}% | "
          f"Sparsity={best['sparsity']:.2f}%")

    plot_gate_distribution(results, best["lambda"])


if __name__ == "__main__":
    main()
