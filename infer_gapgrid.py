"""
Inference script: estimate node IDs from phase / pattern / code images.

Inputs (per image):
  h_phase, v_phase            -- phase images  (grayscale, R channel used)
  colored_traindata           -- pattern image  (G channel used)
  traindata_code              -- code image     (R channel used)

Output:
  nodeid-format PNG per image: R=0, G=id//256, B=id%256

Inference processes the full image directly in a single forward pass (no tiling):
the network's window attention only depends on window_size, not image resolution,
so a model trained at 224x224 can be run on arbitrarily larger images as long as
H, W are divisible by `net.swin_unet.size_divisor` (224 by default). Images that
aren't an exact multiple are reflect-padded up to the next multiple, then cropped
back to the original size after inference -- this avoids the tile-boundary
artifacts that sliding-window inference produces.

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
from scipy import ndimage as ndi
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from datasets.dataset_gapgrid import NUM_CLASSES
from networks.vision_transformer import SwinUnet


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_channel(path: str, ch: int) -> np.ndarray:
    """Load one channel from a PNG (handles both grayscale and RGB)."""
    arr = np.array(Image.open(path))
    return arr if arr.ndim == 2 else arr[:, :, ch]


def make_phase_region_mask(h: np.ndarray, v: np.ndarray,
                           disc_threshold: int = 50, radius: int = 8) -> np.ndarray:
    """
    Segment the image into regions by phase discontinuities in h and v,
    then return mask M: circles of radius `radius` px around each region centroid.

    h, v  : uint8 (H, W)
    returns: uint8 (H, W), 1 inside circles, 0 outside
    """
    H, W = h.shape
    boundary = np.zeros((H, W), dtype=bool)

    for ch in (h.astype(np.float32), v.astype(np.float32)):
        dx = np.abs(np.diff(ch, axis=1))
        boundary[:, :-1] |= dx > disc_threshold
        boundary[:, 1:]  |= dx > disc_threshold
        dy = np.abs(np.diff(ch, axis=0))
        boundary[:-1, :] |= dy > disc_threshold
        boundary[1:, :]  |= dy > disc_threshold

    labels, n_labels = ndi.label(~boundary)

    ys, xs = np.mgrid[0:H, 0:W]
    mask = np.zeros((H, W), dtype=np.uint8)
    for lbl in range(1, n_labels + 1):
        region = labels == lbl
        cy = float(ys[region].mean())
        cx = float(xs[region].mean())
        mask[(ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2] = 1

    return mask


def load_input(data_dir: str, fname: str) -> np.ndarray:
    """Return (H, W, 4) float32 in [0, 1], with phase-region mask applied to code."""
    h = _load_channel(os.path.join(data_dir, 'h_phase',           fname), 0)
    v = _load_channel(os.path.join(data_dir, 'v_phase',           fname), 0)
    c = _load_channel(os.path.join(data_dir, 'colored_traindata',  fname), 1)
    t = _load_channel(os.path.join(data_dir, 'traindata_code',    fname), 0)

    mask = make_phase_region_mask(h, v)
    t = t * mask  # zero code outside region centroids

    return np.stack([h, v, c, t], axis=2).astype(np.float32) / 255.0


def pred_to_nodeid_image(pred: np.ndarray) -> np.ndarray:
    """Encode int32 class-id map as nodeid PNG: R=0, G=id//256, B=id%256."""
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 1] = (pred >> 8) & 0xFF  # G
    rgb[:, :, 2] =  pred       & 0xFF  # B
    return rgb


# ---------------------------------------------------------------------------
# Direct (non-tiled) full-image inference
# ---------------------------------------------------------------------------

def pad_to_multiple(image_4ch: np.ndarray, multiple: int):
    """Reflect-pad image_4ch (H, W, C) so H, W become multiples of `multiple`.

    Returns (padded, orig_H, orig_W).
    """
    H, W = image_4ch.shape[:2]
    pad_h = (-H) % multiple
    pad_w = (-W) % multiple
    if pad_h == 0 and pad_w == 0:
        return image_4ch, H, W
    padded = np.pad(image_4ch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return padded, H, W


def predict_full_image(
    net: torch.nn.Module,
    image_4ch: np.ndarray,
    size_divisor: int = 224,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Run the network directly on the full image in a single forward pass (no
    tiling). The window-attention weights only depend on window_size, not image
    resolution, so this reuses the trained weights as-is. The image is
    reflect-padded up to a multiple of `size_divisor` if necessary, then the
    prediction is cropped back to the original size.

    Returns int32 class-id map of shape (H, W) -- same size as the input.
    """
    padded, H, W = pad_to_multiple(image_4ch, size_divisor)
    t = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    net.eval()
    with torch.no_grad():
        pred = net(t).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    return pred[:H, :W]


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

    if args.indices is not None:
        fnames = [f'{i:06d}.png' for i in args.indices]
    else:
        fnames = sorted(f for f in os.listdir(os.path.join(args.data_dir, 'h_phase'))
                        if f.endswith('.png'))

    size_divisor = net.swin_unet.size_divisor
    print(f'Processing {len(fnames)} images directly (no tiling), '
          f'padded to multiples of {size_divisor}')

    for fname in tqdm(fnames, unit='img'):
        image_4ch = load_input(args.data_dir, fname)
        pred      = predict_full_image(net, image_4ch,
                                       size_divisor=size_divisor, device=device)
        rgb       = pred_to_nodeid_image(pred)
        Image.fromarray(rgb).save(os.path.join(args.output_dir, fname))

    print(f'Done. Predictions saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
