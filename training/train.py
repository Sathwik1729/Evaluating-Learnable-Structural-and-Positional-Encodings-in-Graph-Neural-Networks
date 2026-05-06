import time
import torch
import torch.nn as nn
from .evaluate import evaluate


def train(model, data, config, device, verbose=True):
    """
    Full training loop for node classification.

    Returns:
        best_val_acc: float
        best_test_acc: float  (recorded at epoch of best val)
        history: dict with lists 'train_loss', 'val_acc', 'test_acc'
        train_time: float (seconds)
    """
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': [], 'test_acc': []}
    best_val_acc = 0.0
    best_test_acc = 0.0

    t0 = time.time()

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        history['train_loss'].append(loss.item())
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:>4d} | Loss: {loss.item():.4f} | "
                  f"Val: {val_acc:.4f} | Test: {test_acc:.4f}")

    train_time = time.time() - t0

    if verbose:
        print(f"\nBest Val: {best_val_acc:.4f} | Test @ Best Val: {best_test_acc:.4f} "
              f"| Time: {train_time:.1f}s")

    return best_val_acc, best_test_acc, history, train_time
