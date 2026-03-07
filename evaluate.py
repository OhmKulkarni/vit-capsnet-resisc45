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


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate vit-capsnet-resisc45')
    parser.add_argument('--config',  type=str, default='config.yaml')
    parser.add_argument('--compare', action='store_true',
                        help='Print cross-stage comparison table and exit')
    parser.add_argument('--all',     action='store_true',
                        help='Evaluate all stages that have a best checkpoint')
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[device] CPU")
    return device


def evaluate_stage(config, device, test_loader, class_names):
    mode      = config['mode']
    ckpt_dir  = config['paths']['checkpoints_dir']
    ckpt_path = os.path.join(ckpt_dir, f"best_{mode}.pth")

    if not os.path.exists(ckpt_path):
        print(f"[eval] No checkpoint at {ckpt_path} — skipping {mode}.")
        return

    print(f"\n{'='*60}")
    print(f"  Evaluating: {mode}")
    print(f"{'='*60}")

    # Load model
    # weights_only=False needed because checkpoint contains config dict
    model = CombinedModel(config, num_classes=len(class_names)).to(device)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    print(f"[eval] Loaded checkpoint: {ckpt_path}  "
          f"(epoch={state['epoch']}, val_acc={state['val_acc']:.4f})")

    # Test set inference
    print("[eval] Running test set inference...")
    preds, labels = evaluate_model(model, test_loader, device)

    # Metrics
    summary = compute_metrics(preds, labels, class_names)
    print(f"\n[eval] Test accuracy    : {summary['overall_accuracy']:.4f}")
    print(f"[eval] Macro F1         : {summary['macro_f1']:.4f}")
    print(f"[eval] Weighted F1      : {summary['weighted_f1']:.4f}")

    # Top and bottom 5 classes
    per_class = summary['per_class']
    sorted_cls = sorted(per_class.items(), key=lambda x: x[1]['f1'])
    print(f"\n[eval] 5 weakest classes:")
    for cls, m in sorted_cls[:5]:
        print(f"  {cls:<28}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    print(f"\n[eval] 5 strongest classes:")
    for cls, m in sorted_cls[-5:][::-1]:
        print(f"  {cls:<28}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    # Generate all plots
    print("\n[eval] Generating plots...")
    plot_confusion_matrix(preds, labels, class_names, config, normalize=True)
    plot_per_class_f1(summary, config)
    plot_training_curves(config)

    # Save JSON summary
    save_results_summary(summary, config)

    print(f"\n[eval] All outputs saved to: {config['paths']['results_dir']}/")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    args   = parse_args()
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  vit-capsnet-resisc45 — Evaluation")
    print(f"{'='*60}\n")

    if args.compare:
        compare_stages(config)
        return

    device = get_device()

    print("[data] Loading dataset...")
    _, _, test_loader, class_names = get_dataloaders(config)
    print(f"[data] Test set: {len(test_loader.dataset)} images, "
          f"{len(class_names)} classes\n")

    if args.all:
        all_modes = ['vit_mlp', 'vit_capsule', 'multiscale_capsule']
        for mode in all_modes:
            config['mode'] = mode
            evaluate_stage(config, device, test_loader, class_names)
        compare_stages(config)
    else:
        evaluate_stage(config, device, test_loader, class_names)


if __name__ == '__main__':
    main()