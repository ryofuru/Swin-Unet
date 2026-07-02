import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage, ImageDraw
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pattern_removal import NUM_FRAMES, PatternRemovalDataset, TRAIN_SEQUENCES, VAL_SEQUENCES

SAMPLE_INTERVAL = 50


def _frame_strip(clip_chw, num_frames=NUM_FRAMES):
    """(num_frames*3, H, W) float tensor in [0,1] -> (H, W*num_frames, 3) uint8 image."""
    frames = []
    for i in range(num_frames):
        rgb = clip_chw[i * 3:(i + 1) * 3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        frames.append((rgb * 255).astype(np.uint8))
    return np.concatenate(frames, axis=1)


def _save_sample_images(model, db_val, db_train, epoch_num, snapshot_path):
    """Save Input / GT / Prediction frame strips for one train and one val clip."""
    model.eval()

    def make_block(dataset, idx, label_text):
        sample = dataset[idx]
        image_t = sample['image'].unsqueeze(0).cuda()
        target = sample['target']

        with torch.no_grad():
            pred = torch.sigmoid(model(image_t)).squeeze(0).cpu()

        input_strip = _frame_strip(sample['image'])
        gt_strip = _frame_strip(target)
        pred_strip = _frame_strip(pred)
        H = input_strip.shape[0]
        block = np.concatenate([input_strip, gt_strip, pred_strip], axis=0)

        row_labels = ['Input', 'GT', 'Pred']
        side_w = 50
        side = np.zeros((block.shape[0], side_w, 3), dtype=np.uint8)
        img_side = PILImage.fromarray(side)
        draw = ImageDraw.Draw(img_side)
        for i, name in enumerate(row_labels):
            draw.text((2, i * H + H // 2), name, fill=(255, 255, 255))
        side = np.array(img_side)

        header_h = 20
        header = np.zeros((header_h, side_w + block.shape[1], 3), dtype=np.uint8)
        img_header = PILImage.fromarray(header)
        draw2 = ImageDraw.Draw(img_header)
        draw2.text((2, 3), label_text, fill=(200, 200, 200))
        header = np.array(img_header)

        content = np.concatenate([side, block], axis=1)
        return np.concatenate([header, content], axis=0)

    train_block = make_block(db_train, 0, 'Train')
    val_block = make_block(db_val, 0, 'Val')
    combined = np.concatenate([train_block, val_block], axis=0)

    out_dir = os.path.join(snapshot_path, 'samples')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch_num:04d}.png')
    PILImage.fromarray(combined).save(out_path)
    model.train()
    return out_path


def trainer_pattern_removal(args, model, snapshot_path):
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'),
        filemode='a',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    db_train = PatternRemovalDataset(args.data_dir, TRAIN_SEQUENCES, patch_size=args.img_size, split='train')
    db_val = PatternRemovalDataset(args.data_dir, VAL_SEQUENCES, patch_size=args.img_size, split='val')
    logging.info(f'train: {len(db_train)} clips, val: {len(db_val)} clips')

    train_loader = DataLoader(
        db_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        db_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.max_epochs // 3), gamma=0.5)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    start_epoch = getattr(args, 'start_epoch', 0)
    iter_num = start_epoch * len(train_loader)
    best_val_loss = getattr(args, 'best_val_loss', float('inf'))

    for epoch_num in tqdm(range(start_epoch, start_epoch + args.max_epochs), ncols=70):
        model.train()
        train_loss = 0.0

        for sampled_batch in tqdm(train_loader, desc=f'Train {epoch_num}', leave=False):
            images = sampled_batch['image'].cuda()
            targets = sampled_batch['target'].cuda()

            outputs = torch.sigmoid(model(images))
            loss = l1_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)
            iter_num += 1
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_mse = 0.0

        with torch.no_grad():
            for sampled_batch in tqdm(val_loader, desc=f'Val {epoch_num}', leave=False):
                images = sampled_batch['image'].cuda()
                targets = sampled_batch['target'].cuda()

                outputs = torch.sigmoid(model(images))
                loss = l1_loss(outputs, targets)
                val_loss += loss.item()
                val_mse += nn.functional.mse_loss(outputs, targets).item()

        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        psnr = 10 * math.log10(1.0 / val_mse) if val_mse > 0 else float('inf')

        writer.add_scalar('val/loss', val_loss, epoch_num)
        writer.add_scalar('val/psnr', psnr, epoch_num)
        logging.info(
            f'epoch {epoch_num}: train_loss={train_loss:.4f}  '
            f'val_loss={val_loss:.4f}  val_psnr={psnr:.2f}'
        )

        save_path = os.path.join(snapshot_path, 'best_model.pth' if val_loss < best_val_loss else 'last_model.pth')
        torch.save(model.state_dict(), save_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info(f'  => new best model saved (val_loss={best_val_loss:.4f})')

        is_last = (epoch_num == start_epoch + args.max_epochs - 1)
        if epoch_num % SAMPLE_INTERVAL == 0 or is_last:
            path = _save_sample_images(model, db_val, db_train, epoch_num, snapshot_path)
            logging.info(f'  => sample image saved: {path}')

    writer.close()
    return 'Training Finished!'
