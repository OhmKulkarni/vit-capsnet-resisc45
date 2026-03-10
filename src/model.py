"""
model.py

Defines all model components for vit-capsnet-resisc45.
Three modes are supported, controlled by config['mode']:

    vit_mlp            -> Stage 1: VisionTransformer + MLPClassifier (baseline)
    vit_capsule        -> Stage 2: VisionTransformer + CapsuleNetwork (true dynamic routing)
    multiscale_capsule -> Stage 3: dual-scale VisionTransformer + CapsuleNetwork

Components:
    - PatchEmbedding         : Conv2d patch projection + LayerNorm
    - TransformerEncoderLayer: Single transformer block (attention + feedforward)
    - VisionTransformer      : Full ViT encoder producing a CLS token feature vector
    - MLPClassifier          : Deep MLP head (Stage 1 baseline)
    - PrimaryCapsules        : Converts ViT output into primary capsule vectors
    - CapsuleNetwork         : Dynamic routing by agreement (Sabour et al. 2017)
    - CombinedModel          : Top-level model, routes to correct stage based on config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def squash(x, dim=-1):
    """
    Squash activation (Sabour et al. 2017, eq. 1).
    Maps a capsule vector to length in (0, 1) while preserving direction.

    Args:
        x   : Tensor of capsule vectors (..., capsule_dim)
        dim : Dimension along which to compute the norm (default: last)

    Returns:
        Tensor of same shape with lengths squashed to (0, 1).
    """
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm    = norm_sq.sqrt()
    scale   = norm_sq / (1.0 + norm_sq)
    return scale * (x / (norm + 1e-8))


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and projects each patch
    into a dim-dimensional embedding vector using a Conv2d layer.

    Args:
        image_size  : Height/width of input image (assumed square).
        patch_size  : Height/width of each patch.
        dim         : Output embedding dimension.
    """

    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size {image_size} must be divisible by patch_size {patch_size}"

        self.num_patches = (image_size // patch_size) ** 2
        self.projection  = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.norm        = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.projection(x)          # (B, dim, H/P, W/P)
        x = x.flatten(2)                # (B, dim, num_patches)
        x = x.transpose(1, 2)          # (B, num_patches, dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder Layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder block:
        LayerNorm -> Multi-Head Self-Attention -> residual
        LayerNorm -> Feedforward MLP -> residual

    Args:
        dim     : Embedding dimension.
        heads   : Number of attention heads. Must divide dim evenly.
        mlp_dim : Hidden dimension of the feedforward network.
        dropout : Dropout probability.
    """

    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer Encoder
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Vision Transformer encoder.

    Splits image into patches, projects to embeddings, prepends a learnable
    CLS token, adds positional embeddings, passes through N Transformer layers,
    and returns the CLS token as the image representation.

    A two-layer MLP head (dim -> mlp_dim -> dim) refines the CLS token before
    returning it to the classifier.

    Args:
        image_size  : Height/width of input image.
        patch_size  : Height/width of each patch.
        dim         : Embedding dimension.
        depth       : Number of Transformer encoder layers.
        heads       : Number of attention heads.
        mlp_dim     : Hidden dim of Transformer feedforward and MLP head.
        dropout     : Dropout probability.
    """

    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.dropout     = nn.Dropout(dropout)

        self.encoder     = nn.Sequential(*[
            TransformerEncoderLayer(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm        = nn.LayerNorm(dim)
        self.use_gradient_checkpointing = False  # enabled externally for fine encoder

        self.mlp_head    = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        B = x.shape[0]

        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.dropout(x + self.pos_embed)

        if self.use_gradient_checkpointing and self.training:
            # Recompute activations during backward to save VRAM
            # segments=len(encoder) means one checkpoint per layer
            x = checkpoint_sequential(self.encoder, len(self.encoder), x, use_reentrant=False)
        else:
            x = self.encoder(x)

        x   = self.norm(x)

        cls_out = x[:, 0]
        cls_out = self.mlp_head(cls_out)
        return cls_out


# ---------------------------------------------------------------------------
# Stage 1: MLP Classifier Head (baseline)
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Deep MLP classification head.
    Architecture: dim -> 512 -> dim -> 256 -> 128 -> num_classes
    Each hidden layer: BatchNorm1d -> GELU -> Dropout
    Final layer: plain Linear (raw logits for CrossEntropyLoss)

    Args:
        dim         : Input feature dimension (matches ViT output).
        num_classes : Number of output classes.
    """

    def __init__(self, dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Stage 2 & 3: Capsule Network Head (true dynamic routing)
# ---------------------------------------------------------------------------

class PrimaryCapsules(nn.Module):
    """
    Converts the ViT CLS token feature vector into primary capsule vectors.

    Uses a Linear layer to project the input into
    (num_primary_caps * primary_caps_dim) values, then reshapes into
    capsule vectors and applies the squash nonlinearity.

    Args:
        in_dim           : Input feature dimension (ViT output dim).
        num_primary_caps : Number of primary capsules to produce.
        primary_caps_dim : Dimensionality of each primary capsule vector.
    """

    def __init__(self, in_dim, num_primary_caps, primary_caps_dim):
        super().__init__()
        self.num_primary_caps = num_primary_caps
        self.primary_caps_dim = primary_caps_dim
        self.projection = nn.Linear(in_dim, num_primary_caps * primary_caps_dim)

    def forward(self, x):
        # x: (B, in_dim)
        x = self.projection(x)
        x = x.view(x.size(0), self.num_primary_caps, self.primary_caps_dim)
        return squash(x)


class CapsuleNetwork(nn.Module):
    """
    True Capsule Network with dynamic routing by agreement (Sabour et al. 2017).

    Takes primary capsule vectors and routes them to digit capsules
    (one per class) via iterative dynamic routing.

    The length of each digit capsule vector represents the probability
    that the corresponding class is present in the input.

    Dynamic routing (Algorithm 1 from Sabour et al.):
        1. Initialize routing logits b_ij = 0
        2. For each routing iteration:
            a. c_ij = softmax(b_ij)
            b. s_j  = sum_i(c_ij * u_hat_ij)
            c. v_j  = squash(s_j)
            d. b_ij += u_hat_ij · v_j
        3. Return final v_j

    Args:
        num_primary_caps : Number of input (primary) capsules.
        primary_caps_dim : Dimensionality of primary capsule vectors.
        num_digit_caps   : Number of output capsules = num_classes.
        digit_caps_dim   : Dimensionality of digit capsule vectors.
        num_routing      : Number of routing iterations (default 3).
    """

    def __init__(self, num_primary_caps, primary_caps_dim,
                 num_digit_caps, digit_caps_dim, num_routing=3, dropout=0.2):
        super().__init__()
        self.num_primary_caps = num_primary_caps
        self.num_digit_caps   = num_digit_caps
        self.digit_caps_dim   = digit_caps_dim
        self.num_routing      = num_routing
        self.dropout          = nn.Dropout(p=dropout)

        # Transformation matrices W_ij
        # Shape: (1, num_primary_caps, num_digit_caps, digit_caps_dim, primary_caps_dim)
        self.W = nn.Parameter(
            torch.randn(1, num_primary_caps, num_digit_caps,
                        digit_caps_dim, primary_caps_dim) * 0.1
        )

    def forward(self, u):
        """
        Args:
            u : Primary capsule vectors (B, num_primary_caps, primary_caps_dim)

        Returns:
            v : Digit capsule vectors  (B, num_digit_caps, digit_caps_dim)
        """
        B = u.size(0)

        # Apply dropout to primary capsules before routing to prevent co-adaptation
        u = self.dropout(u)

        # u_expanded: (B, num_primary_caps, 1, primary_caps_dim, 1)
        u_expanded = u.unsqueeze(2).unsqueeze(4)

        # W_expanded: (B, num_primary_caps, num_digit_caps, digit_caps_dim, primary_caps_dim)
        W_expanded = self.W.expand(B, -1, -1, -1, -1)

        # u_hat: (B, num_primary_caps, num_digit_caps, digit_caps_dim)
        u_hat = torch.matmul(W_expanded, u_expanded).squeeze(-1)

        # Detach for routing iterations
        u_hat_detached = u_hat.detach()

        # Initialize routing logits
        b = torch.zeros(B, self.num_primary_caps, self.num_digit_caps, device=u.device)

        # Dynamic routing
        for i in range(self.num_routing):
            c = F.softmax(b, dim=2)                          # (B, num_primary, num_digit)
            u_hat_used = u_hat if i == self.num_routing - 1 else u_hat_detached
            s = (c.unsqueeze(-1) * u_hat_used).sum(dim=1)   # (B, num_digit, digit_dim)
            v = squash(s)                                    # (B, num_digit, digit_dim)

            if i < self.num_routing - 1:
                agreement = (u_hat_detached * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement

        return v



# ---------------------------------------------------------------------------
# Stage 4: Patch-level Primary Capsules
# ---------------------------------------------------------------------------

class PatchPrimaryCapsules(nn.Module):
    """
    Converts a sequence of patch token vectors (B, num_patches, dim) into
    primary capsule vectors (B, num_patches, primary_caps_dim).

    Unlike PrimaryCapsules which collapses a single CLS token via Linear,
    this applies a shared Linear projection to every patch token independently,
    preserving spatial structure for routing. Each patch becomes one capsule,
    giving the routing algorithm spatially-grounded part detectors — closer to
    the original Sabour et al. design where primary capsules came from
    spatially-arranged convolutional feature maps.

    Args:
        in_dim          : Input token dimension (ViT dim).
        primary_caps_dim: Dimensionality of each output capsule vector.
    """

    def __init__(self, in_dim, primary_caps_dim):
        super().__init__()
        self.primary_caps_dim = primary_caps_dim
        self.projection = nn.Linear(in_dim, primary_caps_dim)

    def forward(self, x):
        # x: (B, num_patches, in_dim)
        x = self.projection(x)          # (B, num_patches, primary_caps_dim)
        return squash(x)                # squash each patch capsule independently

# ---------------------------------------------------------------------------
# Top-level Combined Model
# ---------------------------------------------------------------------------

class CombinedModel(nn.Module):
    """
    Top-level model routing to the correct stage based on config['mode'].

    vit_mlp:
        Single ViT encoder -> MLPClassifier
        Output: raw logits (B, num_classes) for CrossEntropyLoss

    vit_capsule:
        Single ViT encoder -> PrimaryCapsules -> CapsuleNetwork
        Output: capsule lengths (B, num_classes) for margin loss

    multiscale_capsule:
        Coarse ViT (patch=16) + Fine ViT (patch=8) run in parallel
        CLS tokens concatenated -> PrimaryCapsules -> CapsuleNetwork
        Output: capsule lengths (B, num_classes) for margin loss

    Args:
        config      : Full config dict loaded from config.yaml.
        num_classes : Overrides config num_classes if provided.
    """

    def __init__(self, config, num_classes=None):
        super().__init__()
        self.mode = config['mode']

        mc  = config['model']
        cc  = config.get('capsule', {})
        num_classes = num_classes or mc['num_classes']

        vit_kwargs = dict(
            image_size = mc['image_size'],
            patch_size = mc['patch_size'],
            dim        = mc['dim'],
            depth      = mc['depth'],
            heads      = mc['heads'],
            mlp_dim    = mc['mlp_dim'],
            dropout    = mc['dropout'],
        )

        if self.mode == 'vit_mlp':
            self.encoder    = VisionTransformer(**vit_kwargs)
            self.classifier = MLPClassifier(mc['dim'], num_classes)

        elif self.mode == 'vit_capsule':
            self.encoder    = VisionTransformer(**vit_kwargs)
            num_primary     = cc.get('primary_caps_channels', 128)
            primary_dim     = cc.get('primary_caps_dim', 8)
            digit_dim       = cc.get('digit_caps_dim', 8)    # reduced from 16
            num_routing     = cc.get('num_routing', 3)
            caps_dropout    = cc.get('caps_dropout', 0.2)
            self.primary    = PrimaryCapsules(mc['dim'], num_primary, primary_dim)
            self.classifier = CapsuleNetwork(num_primary, primary_dim,
                                             num_classes, digit_dim, num_routing,
                                             dropout=caps_dropout)

        elif self.mode == 'multiscale_capsule':
            self.encoder_coarse = VisionTransformer(**vit_kwargs)
            fine_kwargs = {**vit_kwargs,
                           'patch_size': mc['patch_size_fine'],
                           'depth': mc.get('depth_fine', 4)}  # shallower — 1024 tokens is 4x coarse compute
            self.encoder_fine   = VisionTransformer(**fine_kwargs)
            # Gradient checkpointing on fine encoder — read from config (needed on 6GB VRAM)
            self.encoder_fine.use_gradient_checkpointing = mc.get('gradient_checkpoint_fine', True)

            fused_dim    = mc['dim'] * 2
            num_primary  = cc.get('primary_caps_channels', 128)
            primary_dim  = cc.get('primary_caps_dim', 8)
            digit_dim    = cc.get('digit_caps_dim', 8)       # reduced from 16
            num_routing  = cc.get('num_routing', 3)
            caps_dropout = cc.get('caps_dropout', 0.2)
            self.primary    = PrimaryCapsules(fused_dim, num_primary, primary_dim)
            self.classifier = CapsuleNetwork(num_primary, primary_dim,
                                             num_classes, digit_dim, num_routing,
                                             dropout=caps_dropout)
        elif self.mode == 'patch_capsule':
            # Coarse encoder: CLS token for global context (patch=16, full depth)
            self.encoder_coarse = VisionTransformer(**vit_kwargs)
            # Fine encoder: all patch tokens become primary capsules (patch=8, shallow)
            fine_kwargs = {**vit_kwargs,
                           'patch_size': mc['patch_size_fine'],
                           'depth'     : mc.get('depth_fine', 2)}
            self.encoder_fine = VisionTransformer(**fine_kwargs)
            self.encoder_fine.use_gradient_checkpointing = mc.get('gradient_checkpoint_fine', True)

            # num_patches for fine encoder: (image_size / patch_size_fine)^2
            patch_size_fine  = mc['patch_size_fine']
            self.num_patches_fine = (mc['image_size'] // patch_size_fine) ** 2

            primary_dim  = cc.get('primary_caps_dim', 8)
            digit_dim    = cc.get('digit_caps_dim', 8)
            num_routing  = cc.get('num_routing', 3)
            caps_dropout = cc.get('caps_dropout', 0.2)

            # Each of the num_patches_fine patch tokens becomes one primary capsule
            self.primary    = PatchPrimaryCapsules(mc['dim'], primary_dim)

            # Capsule network routes num_patches_fine primary caps -> num_classes digit caps
            self.classifier = CapsuleNetwork(
                self.num_patches_fine, primary_dim,
                num_classes, digit_dim, num_routing,
                dropout=caps_dropout
            )

            # Global context projection: coarse CLS token projected to digit_caps_dim,
            # added to digit capsule vectors before final norm (gives routing global context)
            self.global_ctx = nn.Linear(mc['dim'], num_classes * digit_dim)
            self._digit_dim = digit_dim

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. "
                f"Must be one of: 'vit_mlp', 'vit_capsule', 'multiscale_capsule', 'patch_capsule'."
            )

    def forward(self, x):
        if self.mode == 'vit_mlp':
            features = self.encoder(x)
            return self.classifier(features)

        elif self.mode == 'vit_capsule':
            features = self.encoder(x)
            primary  = self.primary(features)
            caps_out = self.classifier(primary)
            return caps_out.norm(dim=-1)

        elif self.mode == 'multiscale_capsule':
            feat_coarse = self.encoder_coarse(x)
            feat_fine   = self.encoder_fine(x)
            features    = torch.cat([feat_coarse, feat_fine], dim=-1)
            primary     = self.primary(features)
            caps_out    = self.classifier(primary)
            return caps_out.norm(dim=-1)

        elif self.mode == 'patch_capsule':
            B = x.size(0)

            # Coarse CLS token: global scene context
            feat_coarse = self.encoder_coarse(x)     # (B, dim)

            # Fine patch tokens: spatially-grounded part features
            # encoder_fine.forward normally returns cls_out — we need raw patch tokens
            # so we call the encoder internals directly
            xf = self.encoder_fine.patch_embed(x)
            cls_f = self.encoder_fine.cls_token.expand(B, -1, -1)
            xf = torch.cat([cls_f, xf], dim=1)
            xf = self.encoder_fine.dropout(xf + self.encoder_fine.pos_embed)
            if self.encoder_fine.use_gradient_checkpointing and self.training:
                xf = checkpoint_sequential(self.encoder_fine.encoder, len(self.encoder_fine.encoder), xf, use_reentrant=False)
            else:
                xf = self.encoder_fine.encoder(xf)
            xf = self.encoder_fine.norm(xf)
            patch_tokens = xf[:, 1:]                 # (B, num_patches_fine, dim) — drop CLS

            # Each patch token -> primary capsule
            primary  = self.primary(patch_tokens)    # (B, num_patches_fine, primary_caps_dim)

            # Dynamic routing over patch capsules
            caps_out = self.classifier(primary)      # (B, num_classes, digit_caps_dim)

            # Inject global context from coarse CLS: reshape to (B, num_classes, digit_caps_dim)
            # and add to digit capsules before squash+norm
            ctx = self.global_ctx(feat_coarse).view(B, -1, self._digit_dim)
            caps_out = squash(caps_out + ctx)

            return caps_out.norm(dim=-1)