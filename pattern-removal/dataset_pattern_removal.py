import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_FRAMES = 8
FRAME_CHANNELS = 3
IN_CHANS = NUM_FRAMES * FRAME_CHANNELS    # 24
OUT_CHANS = IN_CHANS                       # 24

INPUT_SUBDIR = 'proj_pat0'
GT_SUBDIR = 'pointlight'

# Sequences are split by whole video to avoid frame leakage between train/val.
TRAIN_SEQUENCES = ['sequence_0000', 'sequence_0001']
VAL_SEQUENCES = ['sequence_0002']


def _frame_files(seq_dir, subdir):
    d = os.path.join(seq_dir, subdir)
    return sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))


class PatternRemovalDataset(Dataset):
    """
    8-frame video clip -> 8-frame video clip, RGB color image-to-image regression.

    Input:  NUM_FRAMES consecutive RGB frames from `proj_pat0/`, concatenated on
            the channel axis -> (IN_CHANS, H, W) float32 in [0, 1].
    Target: the corresponding NUM_FRAMES consecutive RGB frames from `pointlight/`,
            concatenated the same way -> (OUT_CHANS, H, W) float32 in [0, 1].

    Clips are sampled with a sliding window (stride 1) within each sequence;
    sequences are never mixed inside one clip.

    Train: one random patch_size x patch_size crop position + one random
           horizontal-flip decision per clip, applied identically to every
           frame of both the input and the target.
    Val:   center crop, no flip.
    """

    def __init__(self, base_dir, sequence_names, patch_size=256, split='train',
                 num_frames=NUM_FRAMES, stride=1):
        self.base_dir = base_dir
        self.patch_size = patch_size
        self.split = split
        self.num_frames = num_frames

        self.clips = []  # list of (seq_dir, [frame_filenames])
        for seq_name in sequence_names:
            seq_dir = os.path.join(base_dir, seq_name)
            frames = _frame_files(seq_dir, INPUT_SUBDIR)
            gt_frames = _frame_files(seq_dir, GT_SUBDIR)
            if frames != gt_frames:
                raise ValueError(f'{seq_name}: {INPUT_SUBDIR}/ and {GT_SUBDIR}/ frame lists differ')
            for start in range(0, len(frames) - num_frames + 1, stride):
                self.clips.append((seq_dir, frames[start:start + num_frames]))

    def __len__(self):
        return len(self.clips)

    def _load_clip(self, seq_dir, filenames):
        inputs = [np.array(Image.open(os.path.join(seq_dir, INPUT_SUBDIR, f)).convert('RGB'))
                  for f in filenames]
        targets = [np.array(Image.open(os.path.join(seq_dir, GT_SUBDIR, f)).convert('RGB'))
                   for f in filenames]
        return inputs, targets  # each: list of (H, W, 3) uint8

    def __getitem__(self, idx):
        seq_dir, filenames = self.clips[idx]
        inputs, targets = self._load_clip(seq_dir, filenames)

        H, W = inputs[0].shape[:2]
        ps = self.patch_size

        if self.split == 'train':
            top = np.random.randint(0, H - ps + 1)
            left = np.random.randint(0, W - ps + 1)
            flip = np.random.random() > 0.5
        else:
            top = (H - ps) // 2
            left = (W - ps) // 2
            flip = False

        def crop_stack(frames):
            chans = []
            for frame in frames:
                patch = frame[top:top + ps, left:left + ps]
                if flip:
                    patch = np.fliplr(patch)
                chans.append(patch.astype(np.float32) / 255.0)
            stacked = np.concatenate(chans, axis=2)  # (ps, ps, num_frames*3)
            return torch.from_numpy(stacked.transpose(2, 0, 1).copy())  # (C, ps, ps)

        image_tensor = crop_stack(inputs)
        target_tensor = crop_stack(targets)
        return {'image': image_tensor, 'target': target_tensor}
