import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def metrics(y, probabilities):
    y, probabilities = np.asarray(y).reshape(-1), np.asarray(probabilities).reshape(-1)
    if len(y) != len(probabilities) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected equal lengths and both binary classes')
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid probabilities')
    labels = probabilities > .5  # Historical strict threshold convention.
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return {'accuracy': float(accuracy_score(y, labels)),
            'precision': float(precision_score(y, labels, zero_division=0)),
            'recall': float(recall_score(y, labels, zero_division=0)),
            'f1': float(f1_score(y, labels, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probabilities)),
            'pr_auc': float(average_precision_score(y, probabilities)),
            'confusion_matrix': dict(zip(['tn', 'fp', 'fn', 'tp'], map(int, [tn, fp, fn, tp])))}


def predict(model, data, device='cpu', criterion=None):
    loader = DataLoader(TensorDataset(torch.from_numpy(data.x),
                       torch.tensor(data.y, dtype=torch.float32).reshape(-1, 1)),
                       batch_size=2048, shuffle=False)
    model.eval()
    probabilities, total_loss = [], 0.
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits')
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            if criterion is not None:
                loss = criterion(logits, y.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite validation loss')
                total_loss += loss.item() * len(x)
    return np.concatenate(probabilities), total_loss / len(data.y)


def create_figures(output, history, best_epoch, y, probabilities):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, ConfusionMatrixDisplay
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Train')
    ax.plot([h['epoch'] for h in history], [h['val_loss'] for h in history], label='Validation')
    ax.axvline(best_epoch, color='black', linestyle='--', label=f'Best epoch: {best_epoch}')
    ax.set(xlabel='Epoch', ylabel='Weighted BCE loss', title='Selected primary MLP trial')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curve.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, probabilities > .5, display_labels=['no', 'yes'], ax=ax, colorbar=False)
    ax.set_title('Primary test confusion matrix (threshold > 0.5)')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    p, r, _ = precision_recall_curve(y, probabilities)
    fig, ax = plt.subplots()
    ax.plot(r, p, label=f'AP = {average_precision_score(y, probabilities):.4f}')
    ax.axhline(np.mean(y), color='gray', linestyle='--', label='Test prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title='Primary test precision-recall curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'precision_recall_curve.png', dpi=160)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(y, probabilities)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y, probabilities):.4f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Primary test ROC curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'roc_curve.png', dpi=160)
    plt.close(fig)

