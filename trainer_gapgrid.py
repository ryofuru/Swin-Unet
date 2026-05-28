import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage, ImageDraw, ImageFont
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_gapgrid import GapGridDataset, IGNORE_INDEX, NUM_CLASSES

SAMPLE_INTERVAL = 50


def _id_to_rgb(id_map):
    """Encode class id as nodeid image format: R=0, G=id//256, B=id%256.

    IGNORE_INDEX pixels are rendered as R=255, G=0, B=0 (matching nodeid dont-care).
    """
    H, W = id_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    valid = id_map != IGNORE_INDEX
    ids = id_map[valid].astype(np.int32)
    rgb[valid, 0] = 0
    rgb[valid, 1] = (ids >> 8) & 0xFF   # G = id // 256
    rgb[valid, 2] = ids & 0xFF           # B = id % 256
    rgb[~valid, 0] = 255                 # dont-care: R=255
    return rgb


def _save_sample_images(model, db_val, db_train, epoch_num, snapshot_path):
    """Save side-by-side visualization for one train and one val sample.

    Columns per sample: h_phase | v_phase | colored_G | traindata_code | GT | Prediction
    GT and Pred are encoded as id = G*256 + B (same as nodeid images).
    Rows: top = train sample, bottom = val sample
    """
    model.eval()

    def make_row(dataset, idx, label_text):
        sample = dataset[idx]
        image_t = sample['image'].unsqueeze(0).cuda()
        label_np = sample['label'].numpy()

        with torch.no_grad():
            pred_np = model(image_t).argmax(dim=1).squeeze(0).cpu().numpy()

        img_np = sample['image'].numpy()  # (4, H, W) float32
        H, W = label_np.shape

        def gray3(ch):
            g = (ch * 255).clip(0, 255).astype(np.uint8)
            return np.stack([g, g, g], axis=2)

        gt_rgb = _id_to_rgb(label_np)
        pred_rgb = _id_to_rgb(pred_np)

        row = np.concatenate(
            [gray3(img_np[0]), gray3(img_np[1]), gray3(img_np[2]), gray3(img_np[3]), gt_rgb, pred_rgb],
            axis=1,
        )

        # Add header bar with column labels
        col_labels = ['h_phase', 'v_phase', 'colored_G', 'code', 'GT', 'Pred']
        header_h = 20
        header = np.zeros((header_h, row.shape[1], 3), dtype=np.uint8)
        img_header = PILImage.fromarray(header)
        draw = ImageDraw.Draw(img_header)
        for i, name in enumerate(col_labels):
            draw.text((i * W + 4, 3), name, fill=(255, 255, 255))
        header = np.array(img_header)

        # Left label (Train / Val)
        side_w = 40
        side = np.zeros((H + header_h, side_w, 3), dtype=np.uint8)
        img_side = PILImage.fromarray(side)
        draw2 = ImageDraw.Draw(img_side)
        draw2.text((2, (H + header_h) // 2), label_text, fill=(200, 200, 200))
        side = np.array(img_side)

        content = np.concatenate([header, row], axis=0)
        return np.concatenate([side, content], axis=1)

    train_row = make_row(db_train, 0, 'Train')
    val_row = make_row(db_val, 0, 'Val')

    # Pad widths to match (should be identical)
    combined = np.concatenate([train_row, val_row], axis=0)

    out_dir = os.path.join(snapshot_path, 'samples')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch_num:04d}.png')
    PILImage.fromarray(combined).save(out_path)
    model.train()
    return out_path


def trainer_gapgrid(args, model, snapshot_path):
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'),
        filemode='a',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    all_indices = list(range(200))
    train_indices = all_indices[:180]
    val_indices = all_indices[180:]

    db_train = GapGridDataset(args.data_dir, train_indices, patch_size=args.img_size, split='train')
    db_val = GapGridDataset(args.data_dir, val_indices, patch_size=args.img_size, split='val')
    logging.info(f'train: {len(db_train)} images, val: {len(db_val)} images')

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

    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    start_epoch = getattr(args, 'start_epoch', 0)
    iter_num = start_epoch * len(train_loader)
    max_iterations = (start_epoch + args.max_epochs) * len(train_loader)
    best_val_loss = getattr(args, 'best_val_loss', float('inf'))

    for epoch_num in tqdm(range(start_epoch, start_epoch + args.max_epochs), ncols=70):
        model.train()
        train_loss = 0.0

        for sampled_batch in tqdm(train_loader, desc=f'Train {epoch_num}', leave=False):
            images = sampled_batch['image'].cuda()
            labels = sampled_batch['label'].cuda()

            outputs = model(images)
            loss = ce_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            iter_num += 1
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sampled_batch in tqdm(val_loader, desc=f'Val {epoch_num}', leave=False):
                images = sampled_batch['image'].cuda()
                labels = sampled_batch['label'].cuda()

                outputs = model(images)
                loss = ce_loss(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                valid = labels != IGNORE_INDEX
                correct += (preds[valid] == labels[valid]).sum().item()
                total += valid.sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total if total > 0 else 0.0

        writer.add_scalar('val/loss', val_loss, epoch_num)
        writer.add_scalar('val/accuracy', accuracy, epoch_num)
        logging.info(
            f'epoch {epoch_num}: train_loss={train_loss:.4f}  '
            f'val_loss={val_loss:.4f}  val_acc={accuracy:.4f}'
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
