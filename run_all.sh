#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RAW_DIR=data/raw
ZIP=$RAW_DIR/MIR-WM811K.zip
if [ ! -f "$RAW_DIR/extracted/MIR-WM811K/Python/WM811K.pkl" ]; then
  mkdir -p "$RAW_DIR"
  if [ ! -f "$ZIP" ]; then
    curl -sS -o "$ZIP" "http://mirlab.org/dataSet/public/MIR-WM811K.zip"
  fi
  unzip -q "$ZIP" -d "$RAW_DIR/extracted"
fi

pip install -q -r requirements.txt
pip install -q --index-url https://download.pytorch.org/whl/cpu torch

cd src
python3 eda.py
python3 train_cnn.py
python3 derived_features.py
python3 augmentation.py
python3 lot_drift.py
python3 lot_split_validation.py
python3 ablation_with_ci.py

echo "Done. See ../results/"
