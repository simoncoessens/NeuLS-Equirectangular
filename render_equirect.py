#!/usr/bin/env python3
"""
Render a full 360×180° equirectangular panorama from a trained PanoModel in train.py.

Example
-------
python render_360.py \
    --ckpt checkpoints/my_exp/last.ckpt \
    --output outputs/my_exp/pano360.png \
    --width 4096 \
    --height 2048 \
    --no-tonemap \
    --device cuda:0
"""
import sys
import os
import math
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ─── Ensure train.py (and its dependencies) are on PYTHONPATH ───────────────
sys.path.insert(0, os.path.dirname(__file__))
from train import PanoModel  # your training script with class PanoModel


@torch.no_grad()
def render_equirectangular(
    model: PanoModel,
    save_to: Path,
    width: int,
    height: int,
    tonemap: bool,
    device: str,
    chunk_size: int = 1_000_000,
):
    """
    Renders a full 360×180° panorama by casting rays in spherical coords.
    Splits into chunks of `chunk_size` rays to avoid OOM.
    """
    model = model.to(device).eval()

    # Create per-pixel longitude/latitude
    lon = torch.linspace(-math.pi, math.pi, width, device=device)
    lat = torch.linspace( math.pi/2, -math.pi/2, height, device=device)
    LON, LAT = torch.meshgrid(lon, lat, indexing="xy")

    # Spherical → Cartesian ray directions
    x = torch.cos(LAT) * torch.cos(LON)
    y = torch.sin(LAT)
    z = torch.cos(LAT) * torch.sin(LON)
    dirs = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

    # All rays originate at camera center
    origins = torch.zeros_like(dirs)

    # Dummy `uv` and `t` tensors (unused in inference path)
    N = dirs.shape[0]
    uv = torch.zeros((N, 2), device=device)
    t  = torch.zeros((N, 1), device=device)

    # Inference in chunks
    rgb_chunks = []
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        ori_chunk = origins[start:end]
        dir_chunk = dirs[start:end]
        t_chunk   = t[start:end]
        uv_chunk  = uv[start:end]

        # model.inference returns raw RGB [M×3]
        rgb_chunk = model.inference(t_chunk, uv_chunk, ori_chunk, dir_chunk, training_phase=1.0)
        rgb_chunks.append(rgb_chunk.cpu())

    rgb = torch.cat(rgb_chunks, dim=0).to(device)  # [width*height, 3]

    # Apply CCM + tonemap or raw CCM only
    if tonemap:
        img_tensor = model.color_and_tone(rgb, height, width)  # [3,H,W]
    else:
        img_tensor = model.color(rgb, height, width)           # [3,H,W]

    # Convert to uint8 H×W×3
    img = img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img8 = (img * 255 + 0.5).astype(np.uint8)

    # Save
    save_to.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8).save(save_to)
    print(f"✅ Saved panorama to {save_to.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Render 360° panorama from train.py's PanoModel")
    p.add_argument("--ckpt",      type=Path, required=True, help="Path to Lightning .ckpt file")
    p.add_argument("--output",    type=Path, required=True, help="Where to save the PNG")
    p.add_argument("--width",     type=int,   default=4096, help="Panorama width (pixels)")
    p.add_argument("--height",    type=int,   default=2048, help="Panorama height (pixels)")
    p.add_argument("--no-tonemap",action="store_true",    help="Skip tone-map; emit raw CCM only")
    p.add_argument("--device",    type=str,   default="cuda", help="cuda | cpu | cuda:0 etc.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"🔄 Loading model from {args.ckpt}")
    model = PanoModel.load_from_checkpoint(args.ckpt, cached_data=None)

    # If you saved without the cached volume, reload it now
    if model.data.rgb_volume is None:
        model.load_volume()

    render_equirectangular(
        model,
        save_to=args.output,
        width=args.width,
        height=args.height,
        tonemap=not args.no_tonemap,
        device=args.device,
    )


if __name__ == "__main__":
    main()
