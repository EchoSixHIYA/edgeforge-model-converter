#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
PY_TAG="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"
VENV_DIR="${VENV_DIR:-/home/$(id -un)/.edgeforge-model-converter-venv-py310}"
"$PYTHON_BIN" -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel 'setuptools<81'
python -m pip install -r requirements.txt

mkdir -p vendor wheels data/tasks
if [[ ! -f vendor/rknn_model_zoo/README.md ]]; then
  rm -rf vendor/rknn_model_zoo
  git clone --depth 1 https://github.com/airockchip/rknn_model_zoo.git vendor/rknn_model_zoo
fi
if [[ ! -f vendor/yolov10/RKNN_README_EN.md ]]; then
  rm -rf vendor/yolov10
  git clone --depth 1 https://github.com/airockchip/yolov10.git vendor/yolov10
fi
declare -A exporter_repos=(
  [yolov5]="https://github.com/airockchip/yolov5.git"
  [yolov6]="https://github.com/airockchip/yolov6.git"
  [yolov7]="https://github.com/airockchip/yolov7.git"
  [yolox]="https://github.com/airockchip/YOLOX.git"
  [yolov8]="https://github.com/airockchip/ultralytics_yolov8.git"
  [yolo11]="https://github.com/airockchip/ultralytics_yolo11.git"
  [retinaface]="https://github.com/biubug6/Pytorch_Retinaface.git"
  [mobilesam]="https://github.com/airockchip/MobileSAM.git"
  [yolo_world]="https://github.com/airockchip/YOLO-World.git"
)
for exporter in "${!exporter_repos[@]}"; do
  if [[ ! -d "vendor/exporters/$exporter/.git" ]]; then
    git clone --depth 1 "${exporter_repos[$exporter]}" "vendor/exporters/$exporter"
  fi
done
if [[ ! -d vendor/exporters/yolo_world/third_party/mmyolo/.git ]]; then
  git clone --depth 1 https://github.com/onuralpszr/mmyolo.git \
    vendor/exporters/yolo_world/third_party/mmyolo
fi
python -m pip install timm==1.0.19 transformers==4.46.3

CLIP_ASSETS="vendor/exporters/clip"
mkdir -p "$CLIP_ASSETS"
if [[ ! -f "$CLIP_ASSETS/tokenizer.json" || ! -f "$CLIP_ASSETS/config.json" ]]; then
  HF_HUB_DISABLE_XET=1 hf download openai/clip-vit-base-patch32 \
    --include config.json preprocessor_config.json tokenizer.json tokenizer_config.json \
    special_tokens_map.json vocab.json merges.txt --local-dir "$CLIP_ASSETS" --max-workers 1
fi

PPOCR_REPO="vendor/exporters/ppocr"
if [[ ! -d "$PPOCR_REPO/.git" ]]; then
  git clone --branch release/2.7 --single-branch --depth 1 --filter=blob:none \
    https://github.com/PaddlePaddle/PaddleOCR.git "$PPOCR_REPO"
fi

PPSEG_REPO="vendor/exporters/ppseg"
if [[ ! -d "$PPSEG_REPO/.git" ]]; then
  git clone --branch release/2.8 --single-branch --depth 1 --filter=blob:none \
    https://github.com/PaddlePaddle/PaddleSeg.git "$PPSEG_REPO"
fi

PPYOLOE_REPO="vendor/exporters/ppyoloe"
if [[ ! -d "$PPYOLOE_REPO/.git" ]]; then
  git clone --branch release/2.5 --single-branch --depth 1 --filter=blob:none --no-checkout \
    https://github.com/PaddlePaddle/PaddleDetection.git "$PPYOLOE_REPO"
  git -C "$PPYOLOE_REPO" sparse-checkout init --cone
  git -C "$PPYOLOE_REPO" sparse-checkout set configs/ppyoloe configs/datasets ppdet tools
  git -C "$PPYOLOE_REPO" checkout release/2.5
fi
if ! grep -q 'rknn_optimize=False' "$PPYOLOE_REPO/ppdet/engine/trainer.py"; then
  git -C "$PPYOLOE_REPO" apply \
    ../../rknn_model_zoo/examples/ppyoloe/patch_for_model_export/ppyoloe_rknn_optimize.patch
fi

if ! ldconfig -p 2>/dev/null | grep -q 'libgomp.so.1'; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libgomp1
  else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgomp1
  fi
fi
PPYOLOE_VENV="/home/$(id -un)/.edgeforge-ppyoloe-export-venv"
python3 -m venv "$PPYOLOE_VENV"
"$PPYOLOE_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$PPYOLOE_VENV/bin/python" -m pip install numpy==1.24.4 \
  paddlepaddle==2.5.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
"$PPYOLOE_VENV/bin/python" -m pip install paddle2onnx==1.0.9 onnx==1.16.0 tqdm typeguard==2.13.3 \
  visualdl opencv-python==4.6.0.66 PyYAML shapely scipy terminaltables Cython pycocotools \
  scikit-learn prettytable filelock

# YAMNet accepts an original TensorFlow SavedModel ZIP. Keep its pinned
# conversion stack isolated from the site's RKNN/PyTorch environment.
YAMNET_VENV="/home/$(id -un)/.edgeforge-yamnet-export-venv"
python3 -m venv "$YAMNET_VENV"
"$YAMNET_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$YAMNET_VENV/bin/python" -m pip install \
  numpy==1.24.3 onnx==1.16.0 onnxruntime==1.17.3 \
  tensorflow==2.13.1 tensorflow_hub==0.16.1 tf2onnx==1.16.1

# YOLO-World's official OpenMMLab exporter requires the MMCV binary built for
# Torch 2.0. Keep it isolated until the final exporter-runtime consolidation.
YOLO_WORLD_VENV="/home/$(id -un)/.edgeforge-yolo-world-export-venv"
python3 -m venv "$YOLO_WORLD_VENV"
"$YOLO_WORLD_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$YOLO_WORLD_VENV/bin/python" -m pip install \
  torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
"$YOLO_WORLD_VENV/bin/python" -m pip install \
  mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
"$YOLO_WORLD_VENV/bin/python" -m pip install \
  mmdet==3.0.0 mmengine==0.10.3 supervision==0.19.0 transformers==4.36.2 \
  onnx==1.16.2 onnxruntime==1.19.2 prettytable
"$YOLO_WORLD_VENV/bin/python" -m pip install --no-build-isolation --no-deps \
  -e vendor/exporters/yolo_world/third_party/mmyolo -e vendor/exporters/yolo_world

# SOPHGO YOLO-World v2 needs the CLIP text encoder exported with Torch 1.13.
# Torch 2.x exports SDPA attention that TPU-MLIR can compile into a degraded
# text tower, so keep this tiny exporter isolated from the OpenMMLab stack.
YOLO_WORLD_CLIP_VENV="/home/$(id -un)/.edgeforge-yolo-world-clip-export-venv"
python3 -m venv "$YOLO_WORLD_CLIP_VENV"
"$YOLO_WORLD_CLIP_VENV/bin/python" -m pip install --upgrade pip setuptools==80.9.0 wheel
"$YOLO_WORLD_CLIP_VENV/bin/python" -m pip install \
  torch==1.13.1+cpu torchvision==0.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
"$YOLO_WORLD_CLIP_VENV/bin/python" -m pip install \
  onnx==1.14.1 onnxsim==0.4.17 onnxruntime==1.16.3 ftfy regex "numpy<2" tqdm requests
"$YOLO_WORLD_CLIP_VENV/bin/python" -m pip install \
  git+https://github.com/openai/CLIP.git

shopt -s nullglob
rknn_wheels=(wheels/rknn_toolkit2*-"$PY_TAG"-*.whl wheels/rknn-toolkit2*-"$PY_TAG"-*.whl)
if ((${#rknn_wheels[@]})); then
  rknn_requirements="requirements-rknn-$PY_TAG.txt"
  if [[ -f "$rknn_requirements" ]]; then
    python -m pip install -r "$rknn_requirements"
  else
    echo "Notice: missing $rknn_requirements; installing only dependencies declared by the RKNN wheel."
  fi
  python -m pip install torch==2.4.0+cpu torchvision==0.19.0+cpu --index-url https://download.pytorch.org/whl/cpu
  python -m pip install "${rknn_wheels[@]}"
else
  echo "Notice: no RKNN-Toolkit2 wheel in wheels/ matches $PY_TAG; the site can start, but real RK conversion requires it."
fi

python -m pip install -e vendor/yolov10
echo "Environment is ready. Start with: scripts/start_wsl.sh"
