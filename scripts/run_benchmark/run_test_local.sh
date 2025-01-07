#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

echo "Running benchmark on test data"
echo "  Make sure to run 'scripts/project/build_all_docker_containers.sh'!"

# generate a unique id
RUN_ID="testrun_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="temp/results/${RUN_ID}"

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -c common/nextflow_helpers/labels_ci.config \
  --id cxg_mouse_pancreas_atlas \
  --input_train_mod1 resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod1.h5ad \
  --input_train_mod2 resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod2.h5ad \
  --input_test_mod1 resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/test_mod1.h5ad \
  --input_test_mod2 resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/test_mod2.h5ad \
  --output_state state.yaml \
  --publish_dir "$publish_dir"
