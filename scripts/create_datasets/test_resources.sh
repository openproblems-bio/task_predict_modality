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

for name in bmmc_cite/normal bmmc_cite/swap bmmc_multiome/normal bmmc_multiome/swap; do
  viash run src/methods/knnr_py/config.vsh.yaml -- \
    --input_train_mod1 $OUTPUT_DIR/openproblems_neurips2021/$name/train_mod1.h5ad \
    --input_train_mod2 $OUTPUT_DIR/openproblems_neurips2021/$name/train_mod2.h5ad \
    --input_test_mod1 $OUTPUT_DIR/openproblems_neurips2021/$name/test_mod1.h5ad \
    --output $OUTPUT_DIR/openproblems_neurips2021/$name/prediction.h5ad

  # pre-train simple_mlp
  rm -r $OUTPUT_DIR/openproblems_neurips2021/$name/models/simple_mlp/
  mkdir -p $OUTPUT_DIR/openproblems_neurips2021/$name/models/simple_mlp/
  viash run src/methods/simple_mlp/train/config.vsh.yaml -- \
    --input_train_mod1 $OUTPUT_DIR/openproblems_neurips2021/$name/train_mod1.h5ad \
    --input_train_mod2 $OUTPUT_DIR/openproblems_neurips2021/$name/train_mod2.h5ad \
    --input_test_mod1 $OUTPUT_DIR/openproblems_neurips2021/$name/test_mod1.h5ad \
    --output $OUTPUT_DIR/openproblems_neurips2021/$name/models/simple_mlp/
done

# only run this if you have access to the openproblems-data bucket
aws s3 sync --profile op \
  resources_test/task_predict_modality \
  s3://openproblems-data/resources_test/task_predict_modality \
  --delete --dryrun
