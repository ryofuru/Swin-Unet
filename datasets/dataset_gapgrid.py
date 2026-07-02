import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

IGNORE_INDEX = -1
NUM_CLASSES = 2389


class GapGridDataset(Dataset):
    """
    Input:  4-channel float32 in [0,1]:
              ch0 = h_phase        (R channel; all channels identical)
              ch1 = v_phase        (R channel; all channels identical)
              ch2 = colored_traindata (G channel)
              ch3 = traindata_code (R channel; all channels identical, 6 discrete values)
    Label:  int64 pixel-wise class id = nodeid_G * 256 + nodeid_B
            IGNORE_INDEX for dont-care pixels:
              - nodeid R == 255
              - traindata_mask is black (all channels 0)

    Train: random crop 224x224 + random horizontal flip.
           Retried up to max_retries times if traindata_mask non-zero fraction < 0.5.
    Val:   center crop 224x224.

    samples: list of (base_dir, file_index) tuples.
    """

    def __init__(self, samples, patch_size=224, split='train', max_retries=20):
        self.samples = samples          # [(base_dir, file_index), ...]
        self.patch_size = patch_size
        self.split = split
        self.max_retries = max_retries

    def __len__(self):
        return len(self.samples)

    def _load(self, base_dir, file_index):
        fname = f'{file_index:06d}.png'
        h = np.array(Image.open(os.path.join(base_dir, 'h_phase', fname)))[:, :, 0]
        v = np.array(Image.open(os.path.join(base_dir, 'v_phase', fname)))[:, :, 0]
        c = np.array(Image.open(os.path.join(base_dir, 'colored_traindata', fname)))[:, :, 1]
        t = np.array(Image.open(os.path.join(base_dir, 'traindata_code', fname)))[:, :, 0]
        nodeid = np.array(Image.open(os.path.join(base_dir, 'nodeid', fname)))
        mask = np.array(Image.open(os.path.join(base_dir, 'traindata_mask', fname)))

        # Dilate dont-care region (R != 0) by 1 pixel (8-connectivity)
        dont_care_src = nodeid[:, :, 0] != 0
        dont_care_dilated = binary_dilation(dont_care_src, structure=np.ones((3, 3), dtype=bool))
        nodeid[dont_care_dilated, 0] = 255
        nodeid[dont_care_dilated, 1] = 0
        nodeid[dont_care_dilated, 2] = 0

        image = np.stack([h, v, c, t], axis=2).astype(np.float32) / 255.0

        label = nodeid[:, :, 1].astype(np.int32) * 256 + nodeid[:, :, 2].astype(np.int32)
        mask_valid = ~np.all(mask == 0, axis=2)   # True where mask is non-zero
        dont_care = (nodeid[:, :, 0] == 255) | (~mask_valid)
        label[dont_care] = IGNORE_INDEX

        return image, label.astype(np.int64), mask_valid

    def __getitem__(self, idx):
        base_dir, file_index = self.samples[idx]
        image, label, mask_valid = self._load(base_dir, file_index)
        H, W = image.shape[:2]
        ps = self.patch_size

        if self.split == 'train':
            for _ in range(self.max_retries):
                top = np.random.randint(0, H - ps + 1)
                left = np.random.randint(0, W - ps + 1)
                if mask_valid[top:top + ps, left:left + ps].mean() >= 0.5:
                    break
        else:
            top = (H - ps) // 2
            left = (W - ps) // 2

        image = image[top:top + ps, left:left + ps].copy()
        label = label[top:top + ps, left:left + ps].copy()

        if self.split == 'train' and np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # (4, H, W)
        label_tensor = torch.from_numpy(label)                      # (H, W)
        return {'image': image_tensor, 'label': label_tensor}


_REQUIRED_SUBDIRS = ('h_phase', 'v_phase', 'colored_traindata', 'traindata_code',
                     'nodeid', 'traindata_mask')


def build_samples(data_dirs, val_fraction=0.1):
    """
    Scan each directory in data_dirs, keep only files present in all required
    subdirectories, split into train/val, and return lists of (base_dir, file_index).
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    train_samples, val_samples = [], []
    for base_dir in data_dirs:
        sets = [set(os.listdir(os.path.join(base_dir, d))) for d in _REQUIRED_SUBDIRS]
        common = sorted(sets[0].intersection(*sets[1:]))
        indices = [int(os.path.splitext(f)[0]) for f in common if f.endswith('.png')]
        n_val = max(1, int(len(indices) * val_fraction))
        train_samples.extend((base_dir, i) for i in indices[:-n_val])
        val_samples.extend((base_dir, i) for i in indices[-n_val:])

    return train_samples, val_samples
