#!/usr/bin/env bash
# You run this. It does not run from tests.
# JasonTStanley/RUGD-Segformer — SegFormer-B5, 25 RUGD classes, safetensors.
# https://huggingface.co/JasonTStanley/RUGD-Segformer
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/weights/rugd-segformer"
mkdir -p "${DEST}"
python -m pip install --user 'huggingface_hub>=0.24'
DEST="${DEST}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

dest = os.environ["DEST"]
snapshot_download(repo_id="JasonTStanley/RUGD-Segformer", local_dir=dest)
print("saved", dest)
print("Do not commit the safetensors. The live node stays YOLOE until an adapter is wired.")
PY
