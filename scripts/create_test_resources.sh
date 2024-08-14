#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

RAW_DATA=resources_test/common
DATASET_DIR=resources_test/task_template

mkdir -p $DATASET_DIR

# process dataset
echo Running process_dataset
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -entry auto \
  --input_states "$RAW_DATA/**/state.yaml" \
  --rename_keys 'input:output_dataset' \
  --settings '{"output_train": "$id/train.h5ad", "output_test": "$id/test.h5ad"}' \
  --publish_dir "$DATASET_DIR" \
  --output_state '$id/state.yaml'

# run one method
viash run src/methods/logistic_regression/config.vsh.yaml -- \
    --input_train $DATASET_DIR/pancreas/train.h5ad \
    --input_test $DATASET_DIR/pancreas/test.h5ad \
    --output $DATASET_DIR/pancreas/denoised.h5ad

# run one metric
viash run src/metrics/accuracy/config.vsh.yaml -- \
    --input_predicition $DATASET_DIR/pancreas/predicted.h5ad \
    --input_solution $DATASET_DIR/pancreas/solution.h5ad \
    --output $DATASET_DIR/pancreas/score.h5ad