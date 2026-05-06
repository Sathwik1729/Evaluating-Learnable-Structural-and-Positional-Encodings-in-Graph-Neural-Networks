import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate(model, data, mask):
    """Return accuracy on the nodes indicated by mask."""
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    return correct / mask.sum().item()


@torch.no_grad()
def evaluate_f1(model, data, mask):
    """Return macro-F1 on the nodes indicated by mask."""
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1).cpu().numpy()
    true = data.y[mask].cpu().numpy()
    return f1_score(true, pred[mask.cpu()], average='macro')
