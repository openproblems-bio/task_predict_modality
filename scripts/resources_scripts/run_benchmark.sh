#!/bin/bash

RUN_ID="run_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="s3://openproblems-data/resources/predict_modality/results/${RUN_ID}"

# only process the 'log_cp10k' datasets
cat > /tmp/params.yaml << HERE
id: predict_modality
input_states: s3://openproblems-data/resources/predict_modality/datasets/**/log_cp10k/state.yaml
rename_keys: 'input_train_mod1:output_train_mod1,input_train_mod2:output_train_mod2,input_test_mod1:output_test_mod1,input_test_mod2:output_test_mod2'
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task_predict_modality.git \
  --revision main_build \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels predict_modality,full