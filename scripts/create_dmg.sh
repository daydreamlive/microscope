#!/bin/bash
set -euo pipefail

APP_NAME="Microscope"
BUILD_DIR="build"
DMG_DIR="build/dmg"
DMG_NAME="${APP_NAME}.dmg"

APP_PATH="${BUILD_DIR}/${APP_NAME}.app"

if [ ! -d "${APP_PATH}" ]; then
    echo "Error: ${APP_PATH} not found. Run 'make app' first."
    exit 1
fi

echo "Creating DMG..."

rm -rf "${DMG_DIR}"
mkdir -p "${DMG_DIR}"

cp -R "${APP_PATH}" "${DMG_DIR}/"
ln -s /Applications "${DMG_DIR}/Applications"

cat > "${DMG_DIR}/README - Models.txt" << 'HEREDOC'
Microscope - Model Setup
========================

Models are NOT included in this distribution (~4GB).

Place models in one of these locations:

  1. ~/Library/Application Support/microscope/models/
  2. Next to Microscope.app in a folder called "models/"

Required files:
  - vocab.json
  - merges.txt
  - text_encoder.mlmodelc/
  - unet_sdxs_512.mlmodelc/     (and/or unet_sd_turbo.mlmodelc/)
  - taesd_encoder_512.mlmodelc/
  - taesd_decoder.mlmodelc/

To convert models, run:
  python scripts/convert_models.py --model sdxs --output-dir models/
HEREDOC

rm -f "${BUILD_DIR}/${DMG_NAME}"
hdiutil create -volname "${APP_NAME}" \
    -srcfolder "${DMG_DIR}" \
    -ov -format UDZO \
    "${BUILD_DIR}/${DMG_NAME}"

rm -rf "${DMG_DIR}"

echo "DMG created: ${BUILD_DIR}/${DMG_NAME}"
