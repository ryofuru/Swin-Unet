"""
Debug visualization for Channel-Swin-Unet: center 224×224 patch inference.

Columns: h_phase | v_phase | colored_G | code(raw) | mask M | code(masked) | Pred
Output:  model_out/gapgrid_channelswin/samples-debug/<fname>_center.png
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from datasets.dataset_gapgrid import NUM_CLASSES
from networks.channel_swin_transformer_unet import ChannelSwinUnet
from infer_gapgrid import _load_channel, make_phase_region_mask


def _id_to_rgb(id_map):
    H, W = id_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    ids = id_map.astype(np.int32)
    rgb[:, :, 1] = (ids >> 8) & 0xFF
    rgb[:, :, 2] = ids & 0xFF
    return rgb


def gray3(ch_float):
    g = (ch_float * 255).clip(0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=2)


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    required=True)
    p.add_argument('--output_dir',  default='model_out/gapgrid_channelswin/samples-debug')
    p.add_argument('--checkpoint',  default='model_out/gapgrid_channelswin/best_model.pth')
    p.add_argument('--img_size',    type=int, default=224)
    p.add_argument('--cfg',         default='configs/channel_swin_tiny_patch4_window7_224_gapgrid.yaml')
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
    net.eval()
    print(f'Loaded: {args.checkpoint}')

    fnames = sorted(f for f in os.listdir(os.path.join(args.data_dir, 'h_phase'))
                    if f.endswith('.png'))
    print(f'Found {len(fnames)} image(s): {fnames}')

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in fnames:
        h_full = _load_channel(os.path.join(args.data_dir, 'h_phase',          fname), 0)
        v_full = _load_channel(os.path.join(args.data_dir, 'v_phase',          fname), 0)
        c_full = _load_channel(os.path.join(args.data_dir, 'colored_traindata', fname), 1)
        t_full = _load_channel(os.path.join(args.data_dir, 'traindata_code',   fname), 0)

        mask_full = make_phase_region_mask(h_full, v_full)
        t_masked_full = t_full * mask_full

        H, W = h_full.shape
        ps = args.img_size
        top  = (H - ps) // 2
        left = (W - ps) // 2

        def crop(arr):
            return arr[top:top + ps, left:left + ps]

        h = crop(h_full).astype(np.float32) / 255.0
        v = crop(v_full).astype(np.float32) / 255.0
        c = crop(c_full).astype(np.float32) / 255.0
        t_raw    = crop(t_full).astype(np.float32) / 255.0
        mask_c   = crop(mask_full).astype(np.float32)
        t_masked = crop(t_masked_full).astype(np.float32) / 255.0

        patch = np.stack([h, v, c, t_masked], axis=0)
        inp = torch.from_numpy(patch).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_np = net(inp).argmax(dim=1).squeeze(0).cpu().numpy()

        pred_rgb = _id_to_rgb(pred_np)

        col_labels = ['h_phase', 'v_phase', 'colored_G', 'code(raw)', 'mask M', 'code(masked)', 'Pred']
        row = np.concatenate(
            [gray3(h), gray3(v), gray3(c),
             gray3(t_raw), gray3(mask_c), gray3(t_masked),
             pred_rgb],
            axis=1)

        header_h = 20
        header = np.zeros((header_h, row.shape[1], 3), dtype=np.uint8)
        img_header = PILImage.fromarray(header)
        draw = ImageDraw.Draw(img_header)
        for i, name in enumerate(col_labels):
            draw.text((i * ps + 4, 3), name, fill=(255, 255, 255))
        header = np.array(img_header)

        side_w = 60
        side = np.zeros((ps + header_h, side_w, 3), dtype=np.uint8)
        img_side = PILImage.fromarray(side)
        draw2 = ImageDraw.Draw(img_side)
        draw2.text((2, (ps + header_h) // 2), fname[:8], fill=(200, 200, 200))
        side = np.array(img_side)

        content = np.concatenate([header, row], axis=0)
        combined = np.concatenate([side, content], axis=1)

        out_name = os.path.splitext(fname)[0] + '_center.png'
        out_path = os.path.join(args.output_dir, out_name)
        PILImage.fromarray(combined).save(out_path)
        print(f'Saved: {out_path}  (center top={top}, left={left})')

    print('Done.')


if __name__ == '__main__':
    main()
