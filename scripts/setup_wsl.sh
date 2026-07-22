#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV_DIR="/home/$(id -un)/.edgeforge-model-converter-venv"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
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
)
for exporter in "${!exporter_repos[@]}"; do
  if [[ ! -d "vendor/exporters/$exporter/.git" ]]; then
    git clone --depth 1 "${exporter_repos[$exporter]}" "vendor/exporters/$exporter"
  fi
done

shopt -s nullglob
rknn_wheels=(wheels/rknn_toolkit2*.whl wheels/rknn-toolkit2*.whl)
if ((${#rknn_wheels[@]})); then
  python -m pip install -r requirements-rknn-cp38.txt
  python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
  python -m pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
  python -m pip install "${rknn_wheels[@]}"
else
  echo "提示: wheels/ 中没有 RKNN-Toolkit2 官方 wheel；网站可启动，但真正转换前需要安装它。"
fi

python -m pip install -e vendor/yolov10
echo "环境已准备。启动: $VENV_DIR/bin/gunicorn -b 0.0.0.0:5000 --workers 1 --threads 8 --timeout 3600 app:app"
