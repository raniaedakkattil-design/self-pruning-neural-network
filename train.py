import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleNet, PrunableLinear, get_sparsity

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(train_data, range(5000))
test_subset = torch.utils.data.Subset(test_data, range(1000))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64)

def compute_sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            loss += gates.sum()
    return loss


def train_model(lambda_val):
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):  
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            cls_loss = criterion(out, y)
            sparsity_loss = compute_sparsity_loss(model)

            loss = cls_loss + lambda_val * sparsity_loss
            loss.backward()
            optimizer.step()

    return model


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    sparsity = get_sparsity(model)

    return acc, sparsity


if __name__ == "__main__":
    lambdas = [0.0001, 0.001, 0.01]
    results = []

    for l in lambdas:
        print(f"\nTraining with lambda={l}")
        model = train_model(l)
        acc, sparsity = evaluate(model)
        print(f"Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")
        results.append((l, acc, sparsity))

    print("\nFinal Results:")
    for r in results:
        print(r)
