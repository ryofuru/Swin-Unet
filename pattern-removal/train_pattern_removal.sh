#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR=${data_dir:-'/home/lab-shared/gitrep/blender-render-tool/_test/medshape-colon-shapes-sequence-output-dist'}
OUT_DIR=${out_dir:-"$REPO_ROOT/model_out/pattern_removal"}
CFG=${cfg:-"$SCRIPT_DIR/configs/swin_tiny_patch4_window7_224_patternremoval.yaml"}
EPOCHS=${epoch_time:-150}
LR=${learning_rate:-1e-4}
BATCH=${batch_size:-8}
IMG_SIZE=${img_size:-256}

echo "Starting training: data=${DATA_DIR} out=${OUT_DIR} epochs=${EPOCHS} lr=${LR} batch=${BATCH}"
python "$SCRIPT_DIR/train_pattern_removal.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --cfg "$CFG" \
    --max_epochs "$EPOCHS" \
    --base_lr "$LR" \
    --batch_size "$BATCH" \
    --img_size "$IMG_SIZE" \
    --pretrained
