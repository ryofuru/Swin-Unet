import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from datasets.dataset_gapgrid import NUM_CLASSES
from networks.channel_swin_transformer_unet import ChannelSwinUnet
from trainer_gapgrid_channelswin import trainer_gapgrid

parser = argparse.ArgumentParser()
parser.add_argument('--data_dirs', type=str, nargs='+', required=True,
                    help='one or more paths to outimages directories')
parser.add_argument('--output_dir', type=str, required=True,
                    help='directory to save checkpoints and logs')
parser.add_argument('--cfg', type=str,
                    default='configs/channel_swin_tiny_patch4_window7_224_gapgrid.yaml',
                    metavar='FILE', help='path to config file')
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size per GPU (smaller than standard Swin due to ×16 attention cost)')
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--deterministic', type=int, default=1)
# keep these for compatibility with get_config / update_config
parser.add_argument('--opts', nargs='+', default=None)
parser.add_argument('--zip', action='store_true')
parser.add_argument('--cache-mode', type=str, default='part',
                    choices=['no', 'full', 'part'])
parser.add_argument('--resume', default='')
parser.add_argument('--accumulation-steps', type=int, default=0)
parser.add_argument('--use-checkpoint', action='store_true')
parser.add_argument('--amp-opt-level', type=str, default='O1',
                    choices=['O0', 'O1', 'O2'])
parser.add_argument('--tag', default='')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--throughput', action='store_true')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint .pth to resume from')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch offset for LR schedule and logging')
parser.add_argument('--best_val_loss', type=float, default=float('inf'),
                    help='best val loss carried over from previous run')

args = parser.parse_args()
config = get_config(args)

if __name__ == '__main__':
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    net = ChannelSwinUnet(config, img_size=args.img_size, num_classes=NUM_CLASSES).cuda()

    if args.checkpoint:
        msg = net.load_state_dict(torch.load(args.checkpoint, map_location='cuda'))
        print(f'Loaded checkpoint: {args.checkpoint}  ({msg})')

    trainer_gapgrid(args, net, args.output_dir)
