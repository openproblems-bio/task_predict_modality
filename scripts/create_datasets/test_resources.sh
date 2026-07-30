#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

RAW_DATA=resources_test/common
OUTPUT_DIR=resources_test/task_predict_modality
DATASET_DIR=$OUTPUT_DIR/openproblems_neurips2021

mkdir -p $OUTPUT_DIR

export NXF_VER=25.10.7

# skip a step when its output exists and no input is newer; set FORCE=1 to
# regenerate everything. usage: up_to_date <output> <input>...
FORCE=${FORCE:-0}
up_to_date() {
  local out=$1
  shift

  if [ "$FORCE" -ne 0 ] || [ ! -e "$out" ]; then
    return 1
  fi

  # a directory output only counts once something has been written into it
  if [ -d "$out" ] && [ -z "$(ls -A "$out" 2>/dev/null)" ]; then
    return 1
  fi

  local input
  for input in "$@"; do
    if [ "$input" -nt "$out" ]; then
      return 1
    fi
  done
}

echo "Preprocess datasets"
RAW_STATES=($RAW_DATA/openproblems_neurips2021/*/state.yaml)
if up_to_date $DATASET_DIR/bmmc_cite/normal/state.yaml "${RAW_STATES[@]}" &&
   up_to_date $DATASET_DIR/bmmc_cite/swap/state.yaml "${RAW_STATES[@]}" &&
   up_to_date $DATASET_DIR/bmmc_multiome/normal/state.yaml "${RAW_STATES[@]}" &&
   up_to_date $DATASET_DIR/bmmc_multiome/swap/state.yaml "${RAW_STATES[@]}"; then
  echo "  already up to date, skipping"
else
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
fi

echo "Run one method"

for name in bmmc_cite/normal bmmc_cite/swap bmmc_multiome/normal bmmc_multiome/swap; do
  STATE=$DATASET_DIR/$name/state.yaml

  echo "Run KNN on $name"
  if up_to_date $DATASET_DIR/$name/prediction.h5ad $STATE; then
    echo "  already up to date, skipping"
  else
    viash run src/methods/knnr_py/config.vsh.yaml -- \
      --input_train_mod1 $DATASET_DIR/$name/train_mod1.h5ad \
      --input_train_mod2 $DATASET_DIR/$name/train_mod2.h5ad \
      --input_test_mod1 $DATASET_DIR/$name/test_mod1.h5ad \
      --output $DATASET_DIR/$name/prediction.h5ad
  fi

  echo "pre-train simple_mlp on $name"
  if up_to_date $DATASET_DIR/$name/models/simple_mlp/ $STATE; then
    echo "  already up to date, skipping"
  else
    rm -rf $DATASET_DIR/$name/models/simple_mlp/
    mkdir -p $DATASET_DIR/$name/models/simple_mlp/
    viash run src/methods/simple_mlp/simple_mlp_train/config.vsh.yaml -- \
      --input_train_mod1 $DATASET_DIR/$name/train_mod1.h5ad \
      --input_train_mod2 $DATASET_DIR/$name/train_mod2.h5ad \
      --input_test_mod1 $DATASET_DIR/$name/test_mod1.h5ad \
      --n_epochs 2 \
      --output $DATASET_DIR/$name/models/simple_mlp/
  fi

  # senkin_tmp is CITE-only
  if [[ "$name" == bmmc_cite/normal ]]; then
    echo "pre-train senkin_tmp on $name"
    if up_to_date $DATASET_DIR/$name/models/senkin_tmp/model.pkl $STATE; then
      echo "  already up to date, skipping"
    else
      mkdir -p $DATASET_DIR/$name/models/senkin_tmp/
      viash run src/methods/senkin_tmp/senkin_tmp_train/config.vsh.yaml -- \
        --input_train_mod1 $DATASET_DIR/$name/train_mod1.h5ad \
        --input_train_mod2 $DATASET_DIR/$name/train_mod2.h5ad \
        --input_test_mod1 $DATASET_DIR/$name/test_mod1.h5ad \
        --lgbm_boost_rounds 50 \
        --lgbm_early_stopping 10 \
        --nn_epochs 2 \
        --output $DATASET_DIR/$name/models/senkin_tmp/model.pkl
    fi
  fi

  echo "pre-train novel on $name"
  if up_to_date $DATASET_DIR/$name/models/novel/ $STATE; then
    echo "  already up to date, skipping"
  else
    rm -rf $DATASET_DIR/$name/models/novel/
    mkdir -p $DATASET_DIR/$name/models/novel/
    viash run src/methods/novel/novel_train/config.vsh.yaml -- \
      --input_train_mod1 $DATASET_DIR/$name/train_mod1.h5ad \
      --input_train_mod2 $DATASET_DIR/$name/train_mod2.h5ad \
      --input_test_mod1 $DATASET_DIR/$name/test_mod1.h5ad \
      --n_epochs 2 \
      --output $DATASET_DIR/$name/models/novel
  fi

  # babel only does ATAC->GEX, which is the multiome swap
  if [[ "$name" == bmmc_multiome/swap ]]; then
    echo "pre-train babel on $name"
    if up_to_date $DATASET_DIR/$name/models/babel/output_model.pkl $STATE; then
      echo "  already up to date, skipping"
    else
      mkdir -p $DATASET_DIR/$name/models/babel/
      viash run src/methods/babel/babel_train/config.vsh.yaml -- \
        --input_train_mod1 $DATASET_DIR/$name/train_mod1.h5ad \
        --input_train_mod2 $DATASET_DIR/$name/train_mod2.h5ad \
        --input_test_mod1 $DATASET_DIR/$name/test_mod1.h5ad \
        --nn_epochs 2 \
        --output $DATASET_DIR/$name/models/babel/output_model.pkl
    fi
  fi

done

# only run this if you have access to the openproblems-data bucket
aws s3 sync --profile op \
  resources_test/task_predict_modality \
  s3://openproblems-data/resources_test/task_predict_modality \
  --delete --dryrun
