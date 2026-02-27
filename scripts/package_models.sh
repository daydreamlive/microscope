#!/bin/bash
set -euo pipefail

MODEL_DIR="${1:-models}"
OUT_DIR="${2:-build}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: $MODEL_DIR not found"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Packaging base models..."
cd "$MODEL_DIR"
zip -r "../$OUT_DIR/models-base.zip" \
    text_encoder.mlmodelc \
    taesd_encoder_512.mlmodelc \
    taesd_decoder.mlmodelc \
    vocab.json \
    merges.txt
cd ..

echo "Packaging SDXS model..."
cd "$MODEL_DIR"
zip -r "../$OUT_DIR/models-sdxs.zip" \
    unet_sdxs_512.mlmodelc
cd ..

if [ -d "$MODEL_DIR/unet_sd_turbo.mlmodelc" ]; then
    echo "Packaging SD-Turbo model..."
    cd "$MODEL_DIR"
    zip -r "../$OUT_DIR/models-sd-turbo.zip" \
        unet_sd_turbo.mlmodelc
    cd ..
fi

echo "Done:"
ls -lh "$OUT_DIR"/models-*.zip
