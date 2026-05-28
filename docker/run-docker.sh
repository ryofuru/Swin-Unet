#!/bin/bash
# Run the Swin-Unet training container.
#
# Usage:
#   ./run-docker.sh                  # interactive bash
#   ./run-docker.sh train            # run train_gapgrid.sh with defaults
#   ./run-docker.sh python foo.py    # arbitrary command
#
# Environment overrides (same as train_gapgrid.sh):
#   data_dir=...  out_dir=...  epoch_time=...  learning_rate=...  batch_size=...
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1; pwd -P)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="swin-unet-gapgrid"
CONTAINER_NAME="swin-unet-train"

# Remove a stale container with the same name (if any)
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Resolve the command to run inside the container
if [ "$1" = "train" ]; then
    CMD="bash /workspace/train_gapgrid.sh"
elif [ $# -ge 1 ]; then
    CMD="$*"
else
    CMD="/bin/bash"
fi

echo "Repo   : $REPO_DIR  -> /workspace"
echo "Command: $CMD"

docker run \
    --name "$CONTAINER_NAME" \
    --rm -it \
    --gpus all \
    --shm-size=8gb \
    --workdir /workspace \
    -e data_dir="${data_dir:-gapgrid-dataset/outimages}" \
    -e out_dir="${out_dir:-model_out/gapgrid}" \
    -e epoch_time="${epoch_time:-150}" \
    -e learning_rate="${learning_rate:-0.01}" \
    -e batch_size="${batch_size:-8}" \
    -e img_size="${img_size:-224}" \
    -v "$REPO_DIR":/workspace \
    "$IMAGE_NAME" \
    bash -c "$CMD"
