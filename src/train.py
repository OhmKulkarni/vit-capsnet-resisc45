"""
train.py

Training loop, loss functions, scheduler, early stopping, and checkpoint
saving for vit-capsnet-resisc45.

Loss function is selected automatically based on config['mode']:
    vit_mlp            -> CrossEntropyLoss
    vit_capsule        -> MarginLoss
    multiscale_capsule -> MarginLoss

MarginLoss (Sabour et al. 2017, eq. 4):
    L_k = T_k * max(0, m+ - ||v_k||)^2
        + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

    Where:
        T_k     = 1 if class k is present, 0 otherwise
        m+      = 0.9  (present class margin)
        m-      = 0.1  (absent class margin)
        lambda  = 0.5  (down-weights absent class loss)
        ||v_k|| = length of digit capsule k (from model output)

Memory safety:
    - Outputs are detached immediately after loss computation for metric tracking
    - torch.cuda.empty_cache() called after train and validation each epoch
    - Gradient flush at epoch end guards against double unscale
    - No Python lists accumulate tensors (only .item() scalars stored)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class MarginLoss(nn.Module):
    """
    Capsule Network margin loss (Sabour et al. 2017, eq. 4).

    Operates on capsule lengths (output of CombinedModel for capsule modes),
    which represent class existence probabilities in (0, 1).

    Args:
        m_plus  : Upper margin for present classes (default 0.9).
        m_minus : Lower margin for absent classes (default 0.1).
        lam     : Weight on absent class loss term (default 0.5).
    """

    def __init__(self, m_plus=0.9, m_minus=0.1, lam=0.5):
        super().__init__()
        self.m_plus  = m_plus
        self.m_minus = m_minus
        self.lam     = lam

    def forward(self, lengths, labels):
        """
        Args:
            lengths : Capsule lengths (B, num_classes) in (0, 1)
            labels  : Ground truth class indices (B,)

        Returns:
            Scalar loss value.
        """
        num_classes  = lengths.size(1)
        T            = F.one_hot(labels, num_classes=num_classes).float()
        present_loss = T       * F.relu(self.m_plus  - lengths) ** 2
        absent_loss  = self.lam * (1 - T) * F.relu(lengths - self.m_minus) ** 2
        return (present_loss + absent_loss).sum(dim=1).mean()


def get_loss_fn(config):
    """
    Returns the appropriate loss function based on config['mode'].

    Args:
        config : Full config dict.

    Returns:
        Loss function (nn.Module).
    """
    if config['mode'] == 'vit_mlp':
        return nn.CrossEntropyLoss()
    lc = config.get('loss', {})
    return MarginLoss(
        m_plus  = lc.get('m_plus', 0.9),
        m_minus = lc.get('m_minus', 0.1),
        lam     = lc.get('margin_loss_lambda', 0.5),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, scaler,
                    device, accumulation_steps, grad_clip):
    """
    Runs one full training epoch with mixed precision and gradient accumulation.

    Memory safety:
        - outputs.detach() used for all metric computation so the computation
          graph is freed immediately after .backward()
        - loss scalar extracted via .item() before accumulation to avoid
          retaining graph references in the history list
        - torch.cuda.empty_cache() called at the end of the epoch

    Args:
        model             : CombinedModel instance.
        loader            : Training DataLoader.
        optimizer         : AdamW optimizer.
        loss_fn           : CrossEntropyLoss or MarginLoss.
        scaler            : torch.cuda.amp.GradScaler for mixed precision.
        device            : torch.device.
        accumulation_steps: Number of steps to accumulate gradients over.
        grad_clip         : Maximum gradient norm for clipping.

    Returns:
        avg_loss : Average loss over the epoch (Python float).
        accuracy : Training accuracy over the epoch (Python float).
    """
    model.train()
    total_loss        = 0.0
    correct           = 0
    total             = 0
    pending_gradients = False   # tracks whether unstepped gradients exist
    optimizer.zero_grad(set_to_none=True)  # set_to_none frees memory faster than zeroing

    for step, (images, labels) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type,
                            dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            outputs = model(images)
            loss    = loss_fn(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()
        pending_gradients = True

        # Extract scalar metrics immediately — detach so graph can be freed
        with torch.no_grad():
            loss_val = loss.item() * accumulation_steps   # unscale the division
            preds    = outputs.detach().argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        total_loss += loss_val

        # Free tensors explicitly to release VRAM before next batch
        del outputs, loss, preds

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending_gradients = False

    # Flush any remaining accumulated gradients from the last partial batch
    if pending_gradients:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Free VRAM fragmentation before validation runs
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(model, loader, loss_fn, device):
    """
    Runs validation over the full validation set.

    Memory safety:
        - torch.no_grad() throughout — no gradients allocated
        - outputs.detach() for metrics (redundant under no_grad but explicit)
        - torch.cuda.empty_cache() at the end

    Args:
        model   : CombinedModel instance.
        loader  : Validation DataLoader.
        loss_fn : CrossEntropyLoss or MarginLoss.
        device  : torch.device.

    Returns:
        avg_loss : Average validation loss (Python float).
        accuracy : Validation accuracy (Python float).
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type,
                                dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                outputs = model(images)
                loss    = loss_fn(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            del outputs, loss, preds

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when validation accuracy stops improving.

    Args:
        patience : Number of epochs to wait without improvement before stopping.
        mode     : 'max' for accuracy (default), 'min' for loss.
    """

    def __init__(self, patience=15, mode='max'):
        self.patience    = patience
        self.mode        = mode
        self.best        = None
        self.counter     = 0
        self.should_stop = False

    def step(self, metric):
        """
        Returns True if training should stop, False otherwise.
        """
        if self.best is None:
            self.best = metric
            return False

        improved = (metric > self.best) if self.mode == 'max' else (metric < self.best)

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Checkpoint saving and loading
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, val_acc, config, is_best=False):
    """
    Saves a checkpoint to the checkpoints directory.

    Saves two files:
        last_<mode>.pth : Always overwritten with the most recent epoch.
        best_<mode>.pth : Only overwritten when val_acc improves.

    Args:
        model     : CombinedModel instance.
        optimizer : AdamW optimizer.
        epoch     : Current epoch number.
        val_acc   : Current validation accuracy.
        config    : Full config dict.
        is_best   : If True, also saves as best_<mode>.pth.
    """
    mode     = config['mode']
    ckpt_dir = config['paths']['checkpoints_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        'epoch'    : epoch,
        'mode'     : mode,
        'val_acc'  : val_acc,
        'model'    : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config'   : config,
    }

    torch.save(state, os.path.join(ckpt_dir, f"last_{mode}.pth"))

    if is_best:
        best_path = os.path.join(ckpt_dir, f"best_{mode}.pth")
        torch.save(state, best_path)
        print(f"  [checkpoint] Saved best -> {best_path}  (val_acc={val_acc:.4f})")


def load_checkpoint(model, optimizer, config, load_best=True):
    """
    Loads a checkpoint for the current mode.

    Args:
        model     : CombinedModel instance (modified in place).
        optimizer : AdamW optimizer (modified in place).
        config    : Full config dict.
        load_best : If True loads best_<mode>.pth, else last_<mode>.pth.

    Returns:
        start_epoch  : Epoch to resume from.
        best_val_acc : Best validation accuracy seen so far.
    """
    mode     = config['mode']
    ckpt_dir = config['paths']['checkpoints_dir']
    prefix   = 'best' if load_best else 'last'
    path     = os.path.join(ckpt_dir, f"{prefix}_{mode}.pth")

    if not os.path.exists(path):
        print(f"  [checkpoint] No checkpoint at {path}, starting fresh.")
        return 0, 0.0

    # Load to CPU first to avoid inflating VRAM with a second copy on GPU
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    print(f"  [checkpoint] Loaded {path}  "
          f"(epoch={state['epoch']}, val_acc={state['val_acc']:.4f})")
    return state['epoch'] + 1, state['val_acc']


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def save_history(history, config):
    """
    Saves training history to a JSON file in the results directory.

    Args:
        history : Dict with keys 'train_loss', 'val_loss',
                  'train_acc', 'val_acc' — each a list of Python floats.
        config  : Full config dict.
    """
    mode        = config['mode']
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{mode}_history.json")
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  [history] Saved -> {path}")