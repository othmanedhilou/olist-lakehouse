#!/usr/bin/env bash
# Downloads the Olist dataset from Kaggle (needs ~/.kaggle/kaggle.json)
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"
kaggle datasets download -d olistbr/brazilian-ecommerce -p "$DEST" --unzip
ls -1 "$DEST"
