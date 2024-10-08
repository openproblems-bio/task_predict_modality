#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

RAW_DATA=resources_test/common
OUTPUT_DIR=resources_test/task_predict_modality

mkdir -p $OUTPUT_DIR

export NXF_VER=22.04.5

echo "Preprocess datasets"
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -entry auto \
  -c common/nextflow_helpers/labels_ci.config \
  --input_states "resources_test/common/openproblems_neurips2021/**/state.yaml" \
  --rename_keys 'input_mod1:output_mod1;input_mod2:output_mod2' \
  --settings '{"output_train_mod1": "$id/train_mod1.h5ad", "output_train_mod2": "$id/train_mod2.h5ad", "output_test_mod1": "$id/test_mod1.h5ad", "output_test_mod2": "$id/test_mod2.h5ad"}' \
  --publish_dir "$OUTPUT_DIR" \
  --output_state '$id/state.yaml'

echo "Run one method"

viash run src/methods/knnr_py/config.vsh.yaml -- \
  --input_train_mod1 $OUTPUT_DIR/openproblems_neurips2021/bmmc_cite/normal/train_mod1.h5ad \
  --input_train_mod2 $OUTPUT_DIR/openproblems_neurips2021/bmmc_cite/normal/train_mod2.h5ad \
  --input_test_mod1 $OUTPUT_DIR/openproblems_neurips2021/bmmc_cite/normal/test_mod1.h5ad \
  --output $OUTPUT_DIR/openproblems_neurips2021/bmmc_cite/normal/prediction.h5ad

viash run src/methods/knnr_py/config.vsh.yaml -- \
  --input_train_mod1 $OUTPUT_DIR//openproblems_neurips2021/bmmc_cite/swap/train_mod1.h5ad \
  --input_train_mod2 $OUTPUT_DIR//openproblems_neurips2021/bmmc_cite/swap/train_mod2.h5ad \
  --input_test_mod1 $OUTPUT_DIR//openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad \
  --output $OUTPUT_DIR//openproblems_neurips2021/bmmc_cite/swap/prediction.h5ad

viash run src/methods/knnr_py/config.vsh.yaml -- \
  --input_train_mod1 $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/normal/train_mod1.h5ad \
  --input_train_mod2 $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/normal/train_mod2.h5ad \
  --input_test_mod1 $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/normal/test_mod1.h5ad \
  --output $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/normal/prediction.h5ad

viash run src/methods/knnr_py/config.vsh.yaml -- \
  --input_train_mod1 $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/swap/train_mod1.h5ad \
  --input_train_mod2 $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/swap/train_mod2.h5ad \
  --input_test_mod1 $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/swap/test_mod1.h5ad \
  --output $OUTPUT_DIR/openproblems_neurips2021/bmmc_multiome/swap/prediction.h5ad

# only run this if you have access to the openproblems-data bucket
aws s3 sync --profile op \
  "$DATASET_DIR" s3://openproblems-data/resources_test/task_predict_modality \
  --delete --dryrun
