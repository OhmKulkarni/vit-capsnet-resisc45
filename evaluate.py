"""
evaluate.py (root entrypoint)

Loads a trained checkpoint and runs full evaluation on the test set.
Generates confusion matrix, per-class F1 chart, training curves, and
results JSON. Optionally compares all trained stages.

Usage:
    python evaluate.py                        # evaluates current mode in config.yaml
    python evaluate.py --config config.yaml   # explicit config path
    python evaluate.py --compare              # print cross-stage comparison table
    python evaluate.py --all                  # evaluate all trained stages sequentially
"""

import os
import argparse
import yaml
import torch

from src.model    import CombinedModel
from src.dataset  import get_dataloaders
from src.evaluate import (
    evaluate_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    plot_per_class_f1,
    save_results_summary,
    compare_stages,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate vit-capsnet-resisc45')
    parser.add_argument('--config',  type=str, default='config.yaml',
                        help='Path to config yaml (default: config.yaml)')
    parser.add_argument('--compare', action='store_true',
                        help='Print cross-stage comparison table and exit')
    parser.add_argument('--all',     action='store_true',
                        help='Evaluate all stages that have a best checkpoint')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[device] CPU")
    return device


# ---------------------------------------------------------------------------
# Single stage evaluation
# ---------------------------------------------------------------------------

def evaluate_stage(config, device, test_loader, class_names):
    """
    Loads best checkpoint for the current config mode and runs full evaluation.

    Args:
        config      : Full config dict (mode already set).
        device      : torch.device.
        test_loader : Test DataLoader.
        class_names : List of class name strings.
    """
    mode     = config['mode']
    ckpt_dir = config['paths']['checkpoints_dir']
    ckpt_path = os.path.join(ckpt_dir, f"best_{mode}.pth")

    if not os.path.exists(ckpt_path):
        print(f"[eval] No checkpoint found at {ckpt_path} — skipping {mode}.")
        return

    print(f"\n{'='*60}")
    print(f"  Evaluating: {mode}")
    print(f"{'='*60}")

    # Load model
    model = CombinedModel(config, num_classes=len(class_names)).to(device)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    print(f"[eval] Loaded checkpoint: {ckpt_path}  "
          f"(epoch={state['epoch']}, val_acc={state['val_acc']:.4f})")

    # Run inference
    print("[eval] Running test set inference...")
    preds, labels = evaluate_model(model, test_loader, device)

    # Metrics
    summary = compute_metrics(preds, labels, class_names)
    print(f"\n[eval] Overall accuracy : {summary['overall_accuracy']:.4f}")
    print(f"[eval] Macro F1         : {summary['macro_f1']:.4f}")
    print(f"[eval] Weighted F1      : {summary['weighted_f1']:.4f}")

    # Bottom 5 classes
    per_class = summary['per_class']
    bottom5 = sorted(per_class.items(), key=lambda x: x[1]['f1'])[:5]
    print(f"\n[eval] 5 weakest classes by F1:")
    for cls, m in bottom5:
        print(f"  {cls:<28}  F1={m['f1']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}")

    # Plots
    print("\n[eval] Generating plots...")
    plot_confusion_matrix(preds, labels, class_names, config, normalize=True)
    plot_per_class_f1(summary, config)
    plot_training_curves(config)

    # Save JSON
    save_results_summary(summary, config)

    # Clean up model from VRAM before next stage
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  vit-capsnet-resisc45 — Evaluation")
    print(f"{'='*60}\n")

    # Compare only — no inference needed
    if args.compare:
        compare_stages(config)
        return

    device = get_device()

    # Load dataset once — shared across all stage evaluations
    print("[data] Loading dataset...")
    _, _, test_loader, class_names = get_dataloaders(config)
    print(f"[data] Test set: {len(test_loader.dataset)} images, "
          f"{len(class_names)} classes\n")

    if args.all:
        # Evaluate every stage that has a checkpoint
        all_modes = ['vit_mlp', 'vit_capsule', 'multiscale_capsule']
        for mode in all_modes:
            config['mode'] = mode
            evaluate_stage(config, device, test_loader, class_names)
        compare_stages(config)

    else:
        # Evaluate just the mode set in config.yaml
        evaluate_stage(config, device, test_loader, class_names)


if __name__ == '__main__':
    main()