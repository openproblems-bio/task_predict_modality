#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

# generate a unique id
RUN_ID="run_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="s3://openproblems-data/resources/task_predict_modality/results/${RUN_ID}"

# write the parameters to file
cat > /tmp/params.yaml << HERE
input_states: s3://openproblems-data/resources/task_predict_modality/datasets/**/state.yaml
rename_keys: 'input_train_mod1:output_train_mod1;input_train_mod2:output_train_mod2;input_test_mod1:output_test_mod1;input_test_mod2:output_test_mod2"
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task_predict_modality.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels task_predict_modality,full