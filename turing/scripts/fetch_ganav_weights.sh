#!/usr/bin/env bash
# You run this. It does not run from tests.
# Official GA-Nav RUGD group-6 folder (Google Drive):
# https://drive.google.com/drive/folders/1PYn_kT0zBGOIRSaO_5Jivaq3itrShiPT
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/weights"
mkdir -p "${DEST}"
python -m pip install --user gdown
# Whole folder. Pick ganav_rugd_6.pth after it lands.
gdown --folder "https://drive.google.com/drive/folders/1PYn_kT0zBGOIRSaO_5Jivaq3itrShiPT" -O "${DEST}/ganav_drive"
echo "Look in ${DEST}/ganav_drive for ganav_rugd_6.pth"
echo "Copy it to ${DEST}/ganav_rugd_6.pth  (gitignored). Do not commit it."
