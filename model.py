import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Standard Kaiming init for weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Initialize gate_scores to a POSITIVE value so sigmoid(score) starts near 1
        # (all gates open at the start — the optimizer learns to close unimportant ones)
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * 2.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)   # values in (0, 1)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_sparsity(model, threshold=1e-2):
    total = zero = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.numel()
            zero  += (gates < threshold).sum().item()
    return 100 * zero / total if total > 0 else 0.0
