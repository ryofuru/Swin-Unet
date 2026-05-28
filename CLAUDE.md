# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swin-Unet is a pure transformer architecture for medical image segmentation (ECCVW 2022). It applies a U-Net structure entirely in the transformer domain: a Swin Transformer encoder with patch merging for downsampling, skip connections, and a symmetric Swin Transformer decoder with patch expansion for upsampling. Unlike CNN-based U-Nets, there are no convolutional layers in the encoder/decoder — all spatial mixing is done via shifted-window self-attention.

## Environment Setup

Python 3.7 is required. Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, torchvision, timm, einops, medpy, SimpleITK, h5py.

Pretrained Swin-T checkpoint must be placed at `pretrained_ckpt/swin_tiny_patch4_window7_224.pth` before training (download from Google Drive, link in README).

## Commands

**Train** (Synapse dataset, 150 epochs, batch 24, lr 0.05):
```bash
sh train.sh
# or explicitly:
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
  --root_path <DATA_DIR> --max_epochs 150 --output_dir <OUT_DIR> \
  --img_size 224 --base_lr 0.05 --batch_size 24
```

**Test/inference**:
```bash
sh test.sh
# or explicitly:
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
  --root_path <DATA_DIR> --output_dir <OUT_DIR> \
  --max_epochs 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

The `train.sh` and `test.sh` scripts accept overrides via environment variables: `epoch_time`, `out_dir`, `cfg`, `data_dir`, `learning_rate`, `img_size`, `batch_size`.

**Prepare dataset lists for custom datasets** (splits nnUNet preprocessed .npz files into per-slice .npz and generates train/val .txt lists):
```bash
python make_dataset_txt.py --name <dataset_name> --split --data .npz --nnunet <nnunet_preprocessed_root>
```

## Architecture

The model has two entry points:
- `networks/vision_transformer.py` — `SwinUnet`: thin wrapper that handles grayscale→RGB repeat and pretrained weight loading. This is what `train.py` and `test.py` instantiate.
- `networks/swin_transformer_unet_skip_expand_decoder_sys.py` — `SwinTransformerSys`: the full encoder-decoder implementation.

**Encoder** (`SwinTransformerSys.forward_features`): `PatchEmbed` (patch size 4, 224→56×56 tokens) → 4 `BasicLayer` stages with `PatchMerging` downsampling between stages. Feature maps at each stage are saved into `x_downsample` for skip connections.

**Decoder** (`SwinTransformerSys.forward_up_features`): 4 `BasicLayer_up` stages with `PatchExpand` upsampling. Skip connections concat encoder features and project back via `concat_back_dim` linear layers.

**Final upsample** (`up_x4`): `FinalPatchExpand_X4` does ×4 spatial expansion, then a 1×1 Conv2d produces logit maps of shape `(B, num_classes, H, W)`.

**Weight loading** (`SwinUnet.load_from`): encoder weights from Swin-T pretrained checkpoint are also copied into the decoder's mirrored layers (encoder layer `i` → decoder layer `3-i`), initializing both encoder and decoder from pretrained weights.

## Training Details

- Loss: 0.4 × CrossEntropy + 0.6 × Dice
- Optimizer: SGD with poly LR decay (`lr * (1 - iter/max_iter)^0.9`)
- Default: 150 epochs, batch 24, base_lr 0.05, img_size 224
- If `batch_size % 6 == 0` and batch_size ≠ 24, lr scales linearly with batch size
- Checkpoints: `best_model.pth` (best val loss) and `last_model.pth` in `--output_dir`
- TensorBoard logs written to `<output_dir>/log`

## Data Format

**Train/val**: `.npz` files with keys `image` (2D slice, HW) and `label` (HW integer mask). For custom datasets, files may use `data`/`seg` keys as fallback.

**Test**: `.npy.h5` volumetric files with keys `image` (DHW) and `label` (DHW). Inference runs slice-by-slice along the depth axis.

Dataset split lists live in `lists/<dataset_name>/train.txt` and `val.txt` (or `test_vol.txt` for Synapse). Each line is a file path or case name relative to `--root_path`.

## Configuration

Config is managed via `yacs` in `config.py`. The YAML at `configs/swin_tiny_patch4_window7_224_lite.yaml` overrides defaults. Key model knobs: `MODEL.SWIN.DEPTHS` (encoder stage depths), `MODEL.SWIN.DECODER_DEPTHS`, `MODEL.SWIN.NUM_HEADS`, `MODEL.SWIN.EMBED_DIM`, `MODEL.SWIN.WINDOW_SIZE`.

The `--n_class` argument (not `--num_classes`) controls the number of output classes at train/test time; `--num_classes` in the parser is overridden by dataset config.

## Reproducibility Note

Results are GPU-type dependent. The paper used Tesla V100. Seed is fixed (default 1234) via `--seed`. Pre-training is critical — both encoder and decoder are initialized from pretrained weights.
