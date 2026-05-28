import os
import numpy as np
import torch
from PIL import Image
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
    """

    def __init__(self, base_dir, indices, patch_size=224, split='train', max_retries=20):
        self.base_dir = base_dir
        self.indices = indices
        self.patch_size = patch_size
        self.split = split
        self.max_retries = max_retries

    def __len__(self):
        return len(self.indices)

    def _load(self, file_index):
        fname = f'{file_index:06d}.png'
        h = np.array(Image.open(os.path.join(self.base_dir, 'h_phase', fname)))[:, :, 0]
        v = np.array(Image.open(os.path.join(self.base_dir, 'v_phase', fname)))[:, :, 0]
        c = np.array(Image.open(os.path.join(self.base_dir, 'colored_traindata', fname)))[:, :, 1]
        t = np.array(Image.open(os.path.join(self.base_dir, 'traindata_code', fname)))[:, :, 0]
        nodeid = np.array(Image.open(os.path.join(self.base_dir, 'nodeid', fname)))
        mask = np.array(Image.open(os.path.join(self.base_dir, 'traindata_mask', fname)))

        image = np.stack([h, v, c, t], axis=2).astype(np.float32) / 255.0

        label = nodeid[:, :, 1].astype(np.int32) * 256 + nodeid[:, :, 2].astype(np.int32)
        mask_valid = ~np.all(mask == 0, axis=2)   # True where mask is non-zero
        dont_care = (nodeid[:, :, 0] == 255) | (~mask_valid)
        label[dont_care] = IGNORE_INDEX

        return image, label.astype(np.int64), mask_valid

    def __getitem__(self, idx):
        image, label, mask_valid = self._load(self.indices[idx])
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

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # (3, H, W)
        label_tensor = torch.from_numpy(label)                      # (H, W)
        return {'image': image_tensor, 'label': label_tensor}
