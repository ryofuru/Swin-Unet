"""
Inference script for Channel-Swin-Unet on gapgrid data.

Identical interface to infer_gapgrid.py but uses ChannelSwinUnet and
defaults to model_out/gapgrid_channelswin/.

Like infer_gapgrid.py, this processes the full image directly in a single
forward pass (no tiling) -- see infer_gapgrid.predict_full_image for details.
"""

import argparse
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from datasets.dataset_gapgrid import NUM_CLASSES
from networks.channel_swin_transformer_unet import ChannelSwinUnet
from infer_gapgrid import _load_channel, make_phase_region_mask, load_input, \
    pred_to_nodeid_image, predict_full_image


def build_arg_parser():
    p = argparse.ArgumentParser(
        description='Gapgrid node-ID inference with Channel-Swin-Unet')
    p.add_argument('--data_dir',    required=True)
    p.add_argument('--output_dir',  required=True)
    p.add_argument('--checkpoint',
                   default='model_out/gapgrid_channelswin/best_model.pth')
    p.add_argument('--indices', type=int, nargs='*', default=None)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--cfg',
                   default='configs/channel_swin_tiny_patch4_window7_224_gapgrid.yaml')
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

    net = ChannelSwinUnet(config, img_size=args.img_size, num_classes=NUM_CLASSES).to(device)
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
        pred = predict_full_image(net, image_4ch, size_divisor=size_divisor, device=device)
        rgb = pred_to_nodeid_image(pred)
        Image.fromarray(rgb).save(os.path.join(args.output_dir, fname))

    print(f'Done. Predictions saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
