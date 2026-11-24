#!/usr/bin/env bash
# Pushes the raw CSV files to the landing container created by Terraform
set -euo pipefail

SRC="${1:-data/raw}"
ACCOUNT="$(terraform -chdir=infra/terraform output -raw storage_account)"

az storage blob upload-batch \
  --account-name "$ACCOUNT" \
  --auth-mode login \
  --destination landing \
  --destination-path olist \
  --source "$SRC" \
  --pattern "*.csv" \
  --overwrite
