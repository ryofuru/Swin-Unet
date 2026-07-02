import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Allow importing config.py / networks/ from the repo root regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from dataset_pattern_removal import OUT_CHANS
from networks.vision_transformer import SwinUnet
from trainer_pattern_removal import trainer_pattern_removal

DEFAULT_DATA_DIR = ('/home/lab-shared/gitrep/blender-render-tool/_test/'
                     'medshape-colon-shapes-sequence-output-dist')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                    help='path to medshape-colon-shapes-sequence-output-dist (contains sequence_XXXX/)')
parser.add_argument('--output_dir', type=str, required=True,
                    help='directory to save checkpoints and logs')
parser.add_argument('--cfg', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'configs', 'swin_tiny_patch4_window7_224_patternremoval.yaml'),
                    metavar='FILE', help='path to config file')
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size per GPU (clips per batch)')
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--img_size', type=int, default=256)
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

    net = SwinUnet(config, img_size=args.img_size, num_classes=OUT_CHANS).cuda()

    if args.pretrained:
        net.load_from(config)
        print(f'Loaded pretrained weights from: {config.MODEL.PRETRAIN_CKPT}')
    elif args.checkpoint:
        msg = net.load_state_dict(torch.load(args.checkpoint, map_location='cuda'))
        print(f'Loaded checkpoint: {args.checkpoint}  ({msg})')

    trainer_pattern_removal(args, net, args.output_dir)
