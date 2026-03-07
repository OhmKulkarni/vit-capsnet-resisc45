"""
inference.py (root entrypoint)

Runs inference on a single image or a directory of images using a trained
checkpoint. Prints predicted class and confidence score.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --mode vit_capsule
    python inference.py --dir   path/to/folder/
    python inference.py --dir   path/to/folder/ --mode multiscale_capsule
"""

import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import CombinedModel


# ---------------------------------------------------------------------------
# Config and args
# ---------------------------------------------------------------------------

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference — vit-capsnet-resisc45')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--image',  type=str, default=None,
                        help='Path to a single image file')
    parser.add_argument('--dir',    type=str, default=None,
                        help='Path to a directory of images')
    parser.add_argument('--mode',   type=str, default=None,
                        help='Override config mode (vit_mlp / vit_capsule / multiscale_capsule)')
    parser.add_argument('--topk',   type=int, default=5,
                        help='Number of top predictions to show (default: 5)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def get_inference_transform(config):
    image_size = config['model']['image_size']
    mean       = config['data']['image_mean']
    std        = config['data']['image_std']
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(config, class_names, device):
    mode      = config['mode']
    ckpt_path = os.path.join(config['paths']['checkpoints_dir'], f"best_{mode}.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            f"Train the model first with: python train.py"
        )

    model = CombinedModel(config, num_classes=len(class_names)).to(device)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    print(f"[inference] Loaded {ckpt_path}  (val_acc={state['val_acc']:.4f})")
    return model


# ---------------------------------------------------------------------------
# Single image inference
# ---------------------------------------------------------------------------

def predict_image(image_path, model, transform, class_names, device, topk=5):
    """
    Runs inference on a single image file.

    Args:
        image_path  : Path to image file.
        model       : Loaded CombinedModel in eval mode.
        transform   : Inference transform.
        class_names : List of class name strings.
        device      : torch.device.
        topk        : Number of top predictions to return.

    Returns:
        List of (class_name, confidence) tuples sorted by confidence descending.
    """
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            outputs = model(tensor)

        # Convert to probabilities
        if outputs.min() >= 0 and outputs.max() <= 1:
            # Capsule mode — outputs are already lengths in (0,1)
            probs = outputs / outputs.sum(dim=1, keepdim=True)
        else:
            # MLP mode — outputs are raw logits
            probs = torch.softmax(outputs, dim=1)

        probs = probs.squeeze(0).cpu().numpy()

    topk = min(topk, len(class_names))
    top_indices = np.argsort(probs)[::-1][:topk]
    return [(class_names[i], float(probs[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.mode:
        config['mode'] = args.mode

    class_names = config['classes']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[inference] Mode: {config['mode']}  |  Device: {device}")

    model     = load_model(config, class_names, device)
    transform = get_inference_transform(config)

    # Collect image paths
    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"[inference] File not found: {args.image}")
            return
        image_paths = [args.image]

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"[inference] Directory not found: {args.dir}")
            return
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_paths = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        print(f"[inference] Found {len(image_paths)} images in {args.dir}\n")

    else:
        print("[inference] Provide --image or --dir. Use --help for usage.")
        return

    # Run inference
    print(f"\n{'='*55}")
    for path in image_paths:
        filename = os.path.basename(path)
        results  = predict_image(path, model, transform, class_names, device, args.topk)

        print(f"\n  {filename}")
        print(f"  {'─'*45}")
        for i, (cls, conf) in enumerate(results):
            bar    = '█' * int(conf * 30)
            marker = ' ◄' if i == 0 else ''
            print(f"  {i+1}. {cls:<28} {conf:.3f}  {bar}{marker}")
    print(f"\n{'='*55}")


if __name__ == '__main__':
    main()