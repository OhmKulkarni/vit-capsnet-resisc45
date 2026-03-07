"""
evaluate.py

Evaluation, per-class metrics, confusion matrix, and cross-stage comparison
for vit-capsnet-resisc45.

Functions:
    evaluate_model      : Full evaluation on test set, returns predictions and labels
    compute_metrics     : Per-class precision, recall, F1, accuracy
    plot_confusion_matrix: Saves a confusion matrix heatmap
    plot_training_curves : Saves loss and accuracy curves from training history
    save_results_summary : Saves per-class metrics to JSON
    compare_stages       : Prints a side-by-side comparison of all trained stages
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, device):
    """
    Runs inference over a DataLoader and collects all predictions and labels.

    Args:
        model  : CombinedModel instance (already loaded with weights).
        loader : DataLoader (typically test_loader).
        device : torch.device.

    Returns:
        all_preds  : numpy array of predicted class indices (N,)
        all_labels : numpy array of ground truth class indices (N,)
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
            images = images.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type,
                                dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                outputs = model(images)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

            del outputs, preds

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels, class_names):
    """
    Computes overall accuracy and per-class precision, recall, F1.

    Args:
        preds       : numpy array of predicted class indices (N,)
        labels      : numpy array of ground truth class indices (N,)
        class_names : List of class name strings.

    Returns:
        summary : Dict with overall accuracy and per-class metrics.
    """
    overall_acc = accuracy_score(labels, preds)

    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for name in class_names:
        per_class[name] = {
            'precision': round(report[name]['precision'], 4),
            'recall'   : round(report[name]['recall'],    4),
            'f1'       : round(report[name]['f1-score'],  4),
            'support'  : int(report[name]['support']),
        }

    summary = {
        'overall_accuracy' : round(overall_acc, 4),
        'macro_f1'         : round(report['macro avg']['f1-score'], 4),
        'weighted_f1'      : round(report['weighted avg']['f1-score'], 4),
        'per_class'        : per_class,
    }

    return summary


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(preds, labels, class_names, config, normalize=True):
    """
    Saves a confusion matrix heatmap to the results directory.

    Args:
        preds       : numpy array of predicted class indices (N,)
        labels      : numpy array of ground truth class indices (N,)
        class_names : List of class name strings.
        config      : Full config dict.
        normalize   : If True, normalizes each row to sum to 1.0.
    """
    mode        = config['mode']
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # 45 classes needs a large figure to be readable
    fig, ax = plt.subplots(figsize=(22, 18))

    sns.heatmap(
        cm,
        annot=False,          # too many cells for annotation at 45x45
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
        linecolor='#e0e0e0',
        vmin=0,
        vmax=1 if normalize else None,
    )

    ax.set_title(f'Confusion Matrix — {mode}', fontsize=16, pad=20)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

    plt.tight_layout()
    path = os.path.join(results_dir, f"{mode}_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [eval] Confusion matrix saved -> {path}")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(config):
    """
    Loads training history JSON and saves loss + accuracy curve plots.

    Args:
        config : Full config dict.
    """
    mode        = config['mode']
    results_dir = config['paths']['results_dir']
    history_path = os.path.join(results_dir, f"{mode}_history.json")

    if not os.path.exists(history_path):
        print(f"  [eval] No history file found at {history_path}, skipping curves.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {mode}', fontsize=14)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   linewidth=2)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   label='Val Acc',   linewidth=2)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, f"{mode}_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [eval] Training curves saved -> {path}")


# ---------------------------------------------------------------------------
# Per-class results bar chart
# ---------------------------------------------------------------------------

def plot_per_class_f1(summary, config):
    """
    Saves a horizontal bar chart of per-class F1 scores, sorted descending.
    Useful for identifying which classes the model struggles with.

    Args:
        summary : Output of compute_metrics().
        config  : Full config dict.
    """
    mode        = config['mode']
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    per_class   = summary['per_class']
    class_names = list(per_class.keys())
    f1_scores   = [per_class[c]['f1'] for c in class_names]

    # Sort by F1 ascending so weakest classes are at the top
    sorted_pairs = sorted(zip(f1_scores, class_names))
    f1_sorted    = [p[0] for p in sorted_pairs]
    names_sorted = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, 14))
    colors  = ['#d62728' if f < 0.7 else '#ff7f0e' if f < 0.85 else '#2ca02c'
               for f in f1_sorted]

    bars = ax.barh(names_sorted, f1_sorted, color=colors, edgecolor='white', height=0.7)

    # Add value labels
    for bar, val in zip(bars, f1_sorted):
        ax.text(min(val + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', ha='left', fontsize=8)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_title(f'Per-Class F1 Scores — {mode}\n'
                 f'Overall Acc: {summary["overall_accuracy"]:.4f}  '
                 f'Macro F1: {summary["macro_f1"]:.4f}',
                 fontsize=12)
    ax.axvline(x=summary['macro_f1'], color='navy', linestyle='--',
               linewidth=1.5, label=f'Macro F1 = {summary["macro_f1"]:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, f"{mode}_per_class_f1.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [eval] Per-class F1 chart saved -> {path}")


# ---------------------------------------------------------------------------
# Save results JSON
# ---------------------------------------------------------------------------

def save_results_summary(summary, config):
    """
    Saves the full metrics summary to a JSON file.

    Args:
        summary : Output of compute_metrics().
        config  : Full config dict.
    """
    mode        = config['mode']
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{mode}_results.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [eval] Results summary saved -> {path}")


# ---------------------------------------------------------------------------
# Cross-stage comparison
# ---------------------------------------------------------------------------

def compare_stages(config):
    """
    Reads results JSON files for all three stages and prints a comparison table.
    Only shows stages that have been trained (results file exists).

    Args:
        config : Full config dict.
    """
    results_dir = config['paths']['results_dir']
    stages      = ['vit_mlp', 'vit_capsule', 'multiscale_capsule']
    labels      = ['Stage 1 — ViT + MLP', 'Stage 2 — ViT + Capsule',
                   'Stage 3 — Dual ViT + Capsule']

    found = []
    for stage, label in zip(stages, labels):
        path = os.path.join(results_dir, f"{stage}_results.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            found.append((label, data))

    if not found:
        print("[compare] No results files found yet.")
        return

    print(f"\n{'='*65}")
    print(f"  Stage Comparison — NWPU-RESISC45")
    print(f"{'='*65}")
    print(f"  {'Stage':<35} {'Acc':>8} {'Macro F1':>10} {'Wtd F1':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
    for label, data in found:
        print(f"  {label:<35} "
              f"{data['overall_accuracy']:>8.4f} "
              f"{data['macro_f1']:>10.4f} "
              f"{data['weighted_f1']:>10.4f}")
    print(f"{'='*65}\n")

    # If all three stages are done, show per-class delta (Stage 2 vs Stage 1)
    if len(found) >= 2:
        s1_data = next((d for l, d in found if 'MLP' in l), None)
        s2_data = next((d for l, d in found if 'Capsule' in l and 'Dual' not in l), None)
        if s1_data and s2_data:
            print("  Per-class F1 delta (Stage 2 - Stage 1), top 10 improvements:")
            deltas = {}
            for cls in s1_data['per_class']:
                if cls in s2_data['per_class']:
                    deltas[cls] = (s2_data['per_class'][cls]['f1']
                                   - s1_data['per_class'][cls]['f1'])
            top = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:10]
            for cls, delta in top:
                bar = '█' * int(abs(delta) * 40)
                sign = '+' if delta >= 0 else '-'
                print(f"    {cls:<28} {sign}{abs(delta):.3f}  {bar}")
            print()