"""
train.py (root entrypoint)

Main training script for vit-capsnet-resisc45.

Orchestrates the full training pipeline:
    1. Load config from config.yaml
    2. Build model (mode determined by config['mode'])
    3. Load dataset from HuggingFace
    4. Train with early stopping, mixed precision, gradient accumulation
    5. Save checkpoints and training history

Usage:
    python train.py                        # uses config.yaml defaults
    python train.py --config config.yaml   # explicit config path
    python train.py --resume               # resume from last_<mode>.pth

Switching stages:
    Edit config.yaml and set mode to one of:
        vit_mlp            -> Stage 1 baseline
        vit_capsule        -> Stage 2 capsule head
        multiscale_capsule -> Stage 3 dual-scale capsule
        patch_capsule      -> Stage 4 patch-level primary capsules
    Then run python train.py
"""

import os
import sys
import json
import argparse
import yaml
import torch
import torch.optim as optim

from src.model   import CombinedModel
from src.dataset import get_dataloaders
from src.evaluate import (
    evaluate_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    plot_per_class_f1,
    save_results_summary,
)
from src.train   import (
    get_loss_fn,
    train_one_epoch,
    validate,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    save_history,
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
    parser = argparse.ArgumentParser(description='Train vit-capsnet-resisc45')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config yaml (default: config.yaml)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last_<mode>.pth checkpoint')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[device] GPU: {name}  ({vram:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("[device] No GPU found, using CPU. Training will be slow.")
    return device


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)
    mode   = config['mode']
    tc     = config['training']

    print(f"\n{'='*60}")
    print(f"  vit-capsnet-resisc45")
    print(f"  Mode: {mode}")
    print(f"{'='*60}\n")

    # Device
    device = get_device()

    # Reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    # Data
    print("[data] Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
    print(f"[data] {len(class_names)} classes, "
          f"{len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val\n")

    # Model
    print("[model] Building model...")
    model = CombinedModel(config, num_classes=len(class_names)).to(device)
    total, trainable = count_parameters(model)
    print(f"[model] Parameters: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable\n")

    # Loss
    loss_fn = get_loss_fn(config).to(device)
    print(f"[loss] Using: {'CrossEntropyLoss' if mode == 'vit_mlp' else 'MarginLoss'}")


    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = tc['learning_rate'],
        weight_decay = tc['weight_decay'],
        betas        = tuple(tc['betas']),
    )

    # Scheduler — cosine annealing over full epoch budget
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = tc['epochs'],
        eta_min = tc['eta_min'],
    )

    # Mixed precision scaler (no-op on CPU)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    # Resume from checkpoint if requested
    start_epoch  = 0
    best_val_acc = 0.0
    if args.resume:
        print("\n[checkpoint] Resuming from last checkpoint...")
        start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, config, load_best=False
        )
        # Advance scheduler to match resumed epoch
        for _ in range(start_epoch):
            scheduler.step()

    # Capsule warmup — freeze ViT encoder for first N epochs so routing
    # can bootstrap with a fixed feature extractor before joint training
    warmup_epochs = tc.get('capsule_warmup_epochs', 0) if mode != 'vit_mlp' else 0
    if warmup_epochs > 0:
        print(f"[train] Capsule warmup: freezing ViT encoder for first {warmup_epochs} epochs")
        if mode == 'vit_capsule':
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif mode in ('multiscale_capsule', 'patch_capsule'):
            for p in model.encoder_coarse.parameters():
                p.requires_grad = False
            for p in model.encoder_fine.parameters():
                p.requires_grad = False

    # Early stopping
    early_stopping = EarlyStopping(patience=tc['patience'], mode='max')
    early_stopping.best = best_val_acc if best_val_acc > 0 else None

    # Training history — load existing if resuming so earlier epochs are preserved
    history_path = os.path.join(
        config['paths']['results_dir'], f"{mode}_history.json"
    )
    if args.resume and os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"[history] Loaded {len(history['train_loss'])} existing epochs from {history_path}")
    else:
        history = {
            'train_loss': [],
            'val_loss'  : [],
            'train_acc' : [],
            'val_acc'   : [],
        }

    print(f"\n[train] Starting training from epoch {start_epoch + 1} / {tc['epochs']}")
    print(f"[train] Batch size: {tc['batch_size']}  "
          f"Accumulation steps: {tc['accumulation_steps']}  "
          f"Effective batch: {tc['batch_size'] * tc['accumulation_steps']}")
    print(f"[train] Early stopping patience: {tc['patience']} epochs\n")

    for epoch in range(start_epoch, tc['epochs']):
        # Unfreeze encoder after warmup period
        if warmup_epochs > 0 and epoch == warmup_epochs:
            print(f"\n[train] Warmup complete — unfreezing ViT encoder for joint training")
            if mode == 'vit_capsule':
                for p in model.encoder.parameters():
                    p.requires_grad = True
            elif mode in ('multiscale_capsule', 'patch_capsule'):
                for p in model.encoder_coarse.parameters():
                    p.requires_grad = True
                for p in model.encoder_fine.parameters():
                    p.requires_grad = True

        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else tc['learning_rate']
        print(f"Epoch {epoch+1:>3} / {tc['epochs']}  |  lr={current_lr:.2e}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler,
            device, tc['accumulation_steps'], tc['grad_clip'],
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # Scheduler step
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Save history after every epoch so crashes don't lose progress
        save_history(history, config)

        # Checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        save_checkpoint(model, optimizer, epoch, val_acc, config, is_best=is_best)

        # Early stopping
        if early_stopping.step(val_acc):
            print(f"\n[train] Early stopping triggered after {epoch+1} epochs. "
                  f"Best val_acc={best_val_acc:.4f}")
            break

    # Save history
    save_history(history, config)

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Mode:         {mode}")
    print(f"  Best val_acc: {best_val_acc:.4f}")
    print(f"  Checkpoint:   checkpoints/best_{mode}.pth")
    print(f"{'='*60}\n")

    # --- Auto-evaluate on test set after training ---
    print("[eval] Running post-training evaluation on test set...")
    ckpt_path = os.path.join(config['paths']['checkpoints_dir'], f"best_{mode}.pth")
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()

    preds, labels_arr = evaluate_model(model, test_loader, device)
    summary = compute_metrics(preds, labels_arr, class_names)

    print(f"\n[eval] Test accuracy : {summary['overall_accuracy']:.4f}")
    print(f"[eval] Macro F1      : {summary['macro_f1']:.4f}")
    print(f"[eval] Weighted F1   : {summary['weighted_f1']:.4f}")

    per_class = summary['per_class']
    sorted_cls = sorted(per_class.items(), key=lambda x: x[1]['f1'])
    print(f"\n[eval] 5 weakest classes:")
    for cls, m in sorted_cls[:5]:
        print(f"  {cls:<28}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    print("\n[eval] Generating plots...")
    plot_confusion_matrix(preds, labels_arr, class_names, config, normalize=True)
    plot_per_class_f1(summary, config)
    plot_training_curves(config)
    save_results_summary(summary, config)
    print(f"[eval] All outputs saved to: {config['paths']['results_dir']}/")


if __name__ == '__main__':
    main()