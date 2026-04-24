import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import SimpleNet, PrunableLinear, get_sparsity

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


train_subset = torch.utils.data.Subset(train_data, range(20000))
test_subset  = torch.utils.data.Subset(test_data,  range(2000))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_subset,  batch_size=128)


def compute_sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            loss += gates.sum()
    return loss


def train_model(lambda_val, epochs=15):
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            cls_loss      = criterion(out, y)
            sparsity_loss = compute_sparsity_loss(model)
            loss = cls_loss + lambda_val * sparsity_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss/len(train_loader):.4f}")

    return model


def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    acc      = 100 * correct / total
    sparsity = get_sparsity(model)
    return acc, sparsity


def plot_gate_distribution(model, lambda_val):
    """Plot histogram of all gate values — should show spike near 0 if pruning works."""
    gates_all = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates_all.append(torch.sigmoid(m.gate_scores).detach().cpu().flatten())
    gates_all = torch.cat(gates_all).numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(gates_all, bins=60, color="steelblue", edgecolor="white")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.title(f"Gate value distribution (λ = {lambda_val})")
    plt.tight_layout()
    fname = f"gate_distribution_lambda_{lambda_val}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved plot: {fname}")


if __name__ == "__main__":
    lambdas = [0.0001, 0.001, 0.01]
    results = []
    best_model, best_lambda = None, None

    for l in lambdas:
        print(f"\nTraining with lambda={l}")
        model = train_model(l)
        acc, sparsity = evaluate(model)
        print(f"  Accuracy: {acc:.2f}%  |  Sparsity: {sparsity:.2f}%")
        results.append((l, acc, sparsity))

        
        if l == 0.001:
            best_model, best_lambda = model, l

    print("\n| Lambda | Accuracy (%) | Sparsity (%) |")
    print("|--------|-------------|-------------|")
    for r in results:
        print(f"| {r[0]}  | {r[1]:.2f}        | {r[2]:.2f}       |")

    # Generate the gate distribution plot for the best model
    if best_model:
        plot_gate_distribution(best_model, best_lambda)
