#!/bin/bash

set -e

echo ">> Downloading resources"

# original command:
# common/sync_resources/sync_resources --delete

# temporary workarounds:
mkdir -p resources_test/common
mkdir -p resources_test/predict_modality

common/sync_resources/sync_resources \
  --output . \
  --delete
