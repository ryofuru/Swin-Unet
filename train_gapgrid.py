import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from datasets.dataset_gapgrid import NUM_CLASSES
from networks.vision_transformer import SwinUnet
from trainer_gapgrid import trainer_gapgrid

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='path to gapgrid-dataset/outimages')
parser.add_argument('--output_dir', type=str, required=True,
                    help='directory to save checkpoints and logs')
parser.add_argument('--cfg', type=str,
                    default='configs/swin_tiny_patch4_window7_224_gapgrid.yaml',
                    metavar='FILE', help='path to config file')
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size per GPU')
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
parser.add_argument('--pretrained', action='store_true',
                    help='initialize encoder+decoder from pretrained Swin-T (via config.MODEL.PRETRAIN_CKPT)')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint .pth to resume from (our own trained weights)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch offset for LR schedule and logging (set automatically when --checkpoint is used)')
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

    net = SwinUnet(config, img_size=args.img_size, num_classes=NUM_CLASSES).cuda()

    if args.pretrained:
        net.load_from(config)
        print(f'Loaded pretrained weights from: {config.MODEL.PRETRAIN_CKPT}')
    elif args.checkpoint:
        msg = net.load_state_dict(torch.load(args.checkpoint, map_location='cuda'))
        print(f'Loaded checkpoint: {args.checkpoint}  ({msg})')

    trainer_gapgrid(args, net, args.output_dir)
