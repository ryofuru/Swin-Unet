#!/bin/bash
# Build the Swin-Unet training Docker image.
# Run from anywhere; this script always uses the repo root as build context.
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1; pwd -P)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="swin-unet-gapgrid"

echo "Build context : $REPO_DIR"
echo "Image name    : $IMAGE_NAME"

docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$REPO_DIR"

echo "Done. Image: $IMAGE_NAME"
