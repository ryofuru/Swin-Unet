#!/bin/bash
DATA_DIR=${data_dir:-'gapgrid-dataset/outimages'}
OUT_DIR=${out_dir:-'model_out/gapgrid'}
CFG=${cfg:-'configs/swin_tiny_patch4_window7_224_gapgrid.yaml'}
EPOCHS=${epoch_time:-150}
LR=${learning_rate:-0.01}
BATCH=${batch_size:-8}
IMG_SIZE=${img_size:-224}

echo "Starting training: data=${DATA_DIR} out=${OUT_DIR} epochs=${EPOCHS} lr=${LR} batch=${BATCH}"
python train_gapgrid.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --cfg "$CFG" \
    --max_epochs "$EPOCHS" \
    --base_lr "$LR" \
    --batch_size "$BATCH" \
    --img_size "$IMG_SIZE"
