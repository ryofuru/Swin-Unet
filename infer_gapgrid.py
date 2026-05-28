"""
Inference script: estimate node IDs from phase / pattern / code images.

Inputs (per image):
  h_phase, v_phase            -- phase images  (grayscale, R channel used)
  colored_traindata           -- pattern image  (G channel used)
  traindata_code              -- code image     (R channel used)

Output:
  nodeid-format PNG per image: R=0, G=id//256, B=id%256

Sliding-window inference tiles the full image with 224x224 patches.
Adjacent tiles overlap at the right/bottom edges; the later tile overwrites.

Usage (direct):
  python infer_gapgrid.py \\
      --data_dir  gapgrid-dataset/outimages \\
      --output_dir predictions \\
      --checkpoint model_out/gapgrid/best_model.pth

Usage (Docker):
  ./infer_gapgrid.sh
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from datasets.dataset_gapgrid import NUM_CLASSES
from networks.vision_transformer import SwinUnet


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_input(data_dir: str, file_index: int) -> np.ndarray:
    """Return (H, W, 4) float32 in [0, 1]."""
    fname = f'{file_index:06d}.png'
    h = np.array(Image.open(os.path.join(data_dir, 'h_phase',          fname)))[:, :, 0]
    v = np.array(Image.open(os.path.join(data_dir, 'v_phase',          fname)))[:, :, 0]
    c = np.array(Image.open(os.path.join(data_dir, 'colored_traindata', fname)))[:, :, 1]
    t = np.array(Image.open(os.path.join(data_dir, 'traindata_code',   fname)))[:, :, 0]
    return np.stack([h, v, c, t], axis=2).astype(np.float32) / 255.0


def pred_to_nodeid_image(pred: np.ndarray) -> np.ndarray:
    """Encode int32 class-id map as nodeid PNG: R=0, G=id//256, B=id%256."""
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 1] = (pred >> 8) & 0xFF  # G
    rgb[:, :, 2] =  pred       & 0xFF  # B
    return rgb


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def _patch_starts(size: int, patch: int) -> list[int]:
    """Non-overlapping starts that cover [0, size) completely."""
    starts = list(range(0, size - patch, patch))
    starts.append(size - patch)          # last patch flush with the edge
    return starts


def predict_full_image(
    net: torch.nn.Module,
    image_4ch: np.ndarray,
    patch_size: int = 224,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Tile the full image with patch_size x patch_size crops and run inference.
    Returns int32 class-id map of shape (H, W).
    """
    H, W = image_4ch.shape[:2]
    pred = np.zeros((H, W), dtype=np.int32)

    net.eval()
    with torch.no_grad():
        for top in _patch_starts(H, patch_size):
            for left in _patch_starts(W, patch_size):
                patch = image_4ch[top:top + patch_size, left:left + patch_size]
                t = (torch.from_numpy(patch.transpose(2, 0, 1))
                         .unsqueeze(0).float().to(device))
                p = net(t).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
                pred[top:top + patch_size, left:left + patch_size] = p

    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Gapgrid node-ID inference with Swin-Unet (4-channel)')
    p.add_argument('--data_dir',    required=True,
                   help='root dir containing h_phase/, v_phase/, … sub-dirs')
    p.add_argument('--output_dir',  required=True,
                   help='directory to write prediction PNGs')
    p.add_argument('--checkpoint',  default='model_out/gapgrid/best_model.pth')
    p.add_argument('--indices', type=int, nargs='*', default=None,
                   help='image indices to process (default: 0-199)')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--cfg', default='configs/swin_tiny_patch4_window7_224_gapgrid.yaml')
    # --- dummy args required by get_config ---
    p.add_argument('--opts',               nargs='+', default=None)
    p.add_argument('--zip',                action='store_true')
    p.add_argument('--cache-mode',         default='part')
    p.add_argument('--resume',             default='')
    p.add_argument('--accumulation-steps', type=int, default=0)
    p.add_argument('--use-checkpoint',     action='store_true')
    p.add_argument('--amp-opt-level',      default='O1')
    p.add_argument('--tag',                default='')
    p.add_argument('--eval',               action='store_true')
    p.add_argument('--throughput',         action='store_true')
    p.add_argument('--batch_size',         type=int, default=0)
    return p


def main():
    args = build_arg_parser().parse_args()
    config = get_config(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build and load model
    net = SwinUnet(config, img_size=args.img_size, num_classes=NUM_CLASSES).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(state)
    print(f'Loaded: {args.checkpoint}  (device={device})')

    os.makedirs(args.output_dir, exist_ok=True)

    indices = args.indices if args.indices is not None else list(range(200))
    n_patches = len(_patch_starts(1200, args.img_size)) ** 2   # informational

    print(f'Processing {len(indices)} images, '
          f'{n_patches} patches each ({args.img_size}x{args.img_size})')

    for idx in tqdm(indices, unit='img'):
        image_4ch = load_input(args.data_dir, idx)
        pred      = predict_full_image(net, image_4ch,
                                       patch_size=args.img_size, device=device)
        rgb       = pred_to_nodeid_image(pred)
        Image.fromarray(rgb).save(os.path.join(args.output_dir, f'{idx:06d}.png'))

    print(f'Done. Predictions saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
