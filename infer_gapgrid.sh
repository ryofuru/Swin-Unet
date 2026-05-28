#!/bin/bash
# Run gapgrid node-ID inference inside the training Docker container.
#
# Environment overrides:
#   checkpoint=...   data_dir=...   output_dir=...   indices="0 1 2 ..."
#
# Examples:
#   ./infer_gapgrid.sh
#   checkpoint=model_out/gapgrid/best_model.pth output_dir=predictions ./infer_gapgrid.sh
#   indices="180 181 182" ./infer_gapgrid.sh    # val images only

CHECKPOINT=${checkpoint:-model_out/gapgrid/best_model.pth}
DATA_DIR=${data_dir:-gapgrid-dataset/outimages}
OUTPUT_DIR=${output_dir:-predictions}
INDICES=${indices:-}   # empty = all 200 images

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1; pwd -P)"

INDICES_ARG=""
if [ -n "$INDICES" ]; then
    INDICES_ARG="--indices $INDICES"
fi

echo "checkpoint : $CHECKPOINT"
echo "data_dir   : $DATA_DIR"
echo "output_dir : $OUTPUT_DIR"
echo "indices    : ${INDICES:-all}"

docker run --name swin-unet-infer \
    --rm --gpus all --workdir /workspace \
    -v "$SCRIPT_DIR":/workspace \
    swin-unet-gapgrid \
    python infer_gapgrid.py \
        --data_dir    "$DATA_DIR" \
        --output_dir  "$OUTPUT_DIR" \
        --checkpoint  "$CHECKPOINT" \
        $INDICES_ARG
