#!/bin/bash
# ── Download face-api.js model weights for browser-side face detection ────────
# Uses @vladmandic/face-api (more up-to-date than original face-api.js)
# Models are served as static files from frontend/public/models/

set -e

MODEL_DIR="./frontend/public/models"
BASE_URL="https://raw.githubusercontent.com/vladmandic/face-api/master/model"

echo "🔽 Creating model directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

MODELS=(
  # SSD MobileNet v1 — fast face detection
  "ssd-mobilenetv1-model-weights_manifest.json"
  "ssd-mobilenetv1-model-shard1"
  "ssd-mobilenetv1-model-shard2"

  # FaceNet 68-point landmarks
  "face_landmark_68_model-weights_manifest.json"
  "face_landmark_68_model-shard1"
  "face_landmark_68_tiny_model-weights_manifest.json"
  "face_landmark_68_tiny_model-shard1"

  # Face recognition (128-d descriptors compatible)
  "face_recognition_model-weights_manifest.json"
  "face_recognition_model-shard1"
  "face_recognition_model-shard2"

  # Age + gender (optional)
  "age_gender_model-weights_manifest.json"
  "age_gender_model-shard1"
  "age_gender_model-shard2"
)

echo ""
for MODEL in "${MODELS[@]}"; do
    DEST="$MODEL_DIR/$MODEL"
    if [ -f "$DEST" ]; then
        echo "  ✓ Already exists: $MODEL"
        continue
    fi
    echo "  ⬇ Downloading: $MODEL"
    curl -fsSL "$BASE_URL/$MODEL" -o "$DEST" || echo "  ⚠ Failed: $MODEL (may not exist at this path)"
done

echo ""
echo "✅ Model download complete!"
echo "   Models stored in: $MODEL_DIR"
echo ""
echo "   To verify: ls -lh $MODEL_DIR/"
echo ""
echo "   Usage in JS:"
echo "   import * as faceapi from '@vladmandic/face-api';"
echo "   await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');"
