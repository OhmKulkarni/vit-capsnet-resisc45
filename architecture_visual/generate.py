"""
architecture_visual/generate.py

Generates architecture diagrams for all three model stages using torchview.

Usage:
    cd architecture_visual
    python generate.py

Requirements:
    pip install torchview graphviz
    Install Graphviz system binary: https://graphviz.org/download/
    Add Graphviz to PATH (e.g. C:\Program Files\Graphviz\bin on Windows)

Outputs:
    architecture_visual/stage1_vit_mlp.png
    architecture_visual/stage2_vit_capsule.png
    architecture_visual/stage3_multiscale_capsule.png
"""

import os
import sys
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torchview import draw_graph
from src.model import CombinedModel


def generate(config, mode, output_name):
    config['mode'] = mode
    model = CombinedModel(config)
    model.eval()

    graph = draw_graph(
        model,
        input_size=(1, 3, 256, 256),
        expand_nested=True,
        hide_inner_tensors=True,
        hide_module_functions=False,
        roll=True,
        device='cpu',
    )

    # Set DPI after graph is built
    graph.visual_graph.attr(dpi='150')

    out_path = os.path.join(os.path.dirname(__file__), output_name)
    graph.visual_graph.render(out_path, format='png', cleanup=True)
    print(f"  Saved: {out_path}.png")


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Generating architecture diagrams...\n")

    stages = [
        ('vit_mlp',            'stage1_vit_mlp'),
        ('vit_capsule',        'stage2_vit_capsule'),
        ('multiscale_capsule', 'stage3_multiscale_capsule'),
    ]

    for mode, name in stages:
        print(f"  [{mode}]")
        try:
            generate(config, mode, name)
        except Exception as e:
            print(f"  Failed: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()