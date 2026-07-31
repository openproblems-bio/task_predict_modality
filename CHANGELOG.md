# task_predict_modality 0.2.0

## NEW FUNCTIONALITY

* Added `src/utils/exit_codes.py`, so components can mark themselves non-applicable for a dataset (PR #32).

## MINOR CHANGES

* `file_train_mod1`, `file_train_mod2`, `file_test_mod1`, `file_test_mod2`: Declare `uns["modality"]`, which `process_dataset` already writes and which six methods and the `run_benchmark` workflow already read (PR #29).

* `mse`: Write the unbounded maximum as `"+.inf"` rather than `"+inf"`, which is the literal the metric schema accepts (PR #31).

* `cellmapper_linear`: Write the unmasked variants as `mask_var: null` rather than `mask_var: None`, which YAML reads as the string `"None"` and which would resolve to `adata.var["None"]` (PR #38).

* `comp_method`: Run `check_config.py` as part of the component tests, so method metadata is validated like control methods and metrics already are (PR #37).

* `knnr_py`, `knnr_r`, `lm`, `guanlab_dengkw_pm`: Move `documentation_url` and `repository_url` out of `info` and into the top-level `links` (PR #37).

* `novel`, `simple_mlp`: Add a placeholder Nextflow resource label. Viash renders no process for a `nextflow_script` component, so the label is inert -- it is only there because `check_config` requires one. Can be removed once openproblems-bio/core#41 is released (PR #37).

* `correlation`: Read the paired correlations off proxyC's sparse diagonal instead of `diag(dynutils::calculate_similarity(...))`, which densified an `n_features^2` matrix first. Scores are unchanged (PR #36).

* `novel`: Drop the first of the two identical validation passes in `train_and_valid()`. Its result was never read, so it cost one full pass over the validation set per epoch, 100 epochs per run (PR #41).

* `correlation`: `overall_pearson` and `overall_spearman` are a single correlation of the flattened matrices, not a mean of correlations -- the descriptions said the latter. Also spelled out the zero-variance convention on all six metrics (PR #43).

* Point the `## VIASH START` blocks at files that exist. Several still referenced the openproblems-v2 monorepo layout or the pre-`normal/`-`swap/` resource layout, so running a script directly for debugging failed on the first read. Also added the missing `meta` to `knnr_r`'s block and replaced the borrowed `--id cxg_mouse_pancreas_atlas` in `run_test_local.sh` (PR #45).

* `file_test_mod2`: Label this file "Solution" and say in the description that only the metrics and control methods receive it. It holds the ground truth, but read like just another input (PR #49).

* `run_benchmark`: Emit one dataset metadata entry per dataset by de-duplicating on `dataset_id`, rather than by keeping only the `log_cp10k` states. The old filter emitted nothing at all if a dataset ever arrived under a different normalization (PR #48).

* `file_test_mod2`: Declare `uns["normalization_id"]`, which `run_benchmark` reads off this file to decide which method to run on which dataset (PR #30).

* `lm`: Drop the unused `n_cores` and ask for `lowcpu` rather than `highcpu`. The per-gene loop is `pbapply::pblapply()` without a cluster, so it has always run on one core (PR #42).

* `novel_train`: Store only the metadata `novel_predict` needs in the model artifact, instead of a full copy of `train_mod2` (PR #47).

* `lm`: Fit every column of mod2 in one `solve()` of the normal equations instead of calling `RcppArmadillo::fastLm()` once per column. Every column shares the same design matrix, so the old loop redid an `n x n_pcs` decomposition for each of the ~229k ATAC peaks. Predictions are unchanged; `RcppArmadillo` and `pbapply` are no longer needed (PR #58).

* `novel_predict`, `simple_mlp_predict`, `ss_opm_predict`, `babel_predict`: Ask for `midgpu` rather than `gpu`. All four peak below 4 GB, so they fit the `de.NBI GPU T4 medium` flavour and no longer occupy a whole large node. The train steps stay on `gpu` (PR #58).

* `knnr_py`, `knnr_r`, `cellmapper_linear`: Ask for `highmem` rather than `midmem`, and `lowcpu` rather than `midcpu` for the latter two. `knnr_py` peaks at 82 GB and `cellmapper_linear` at 68 GB against a 50 GB request, and both were OOM-killed on the larger datasets (PR #58).

* `solution`, `zeros`: Ask for `lowmem` rather than `midmem` -- they use 0.5 GB and 11 GB of the 50 GB they asked for (PR #58).

## BUG FIXES

* `guanlab_dengkw_pm`: Restore the consensus scheme of the original submission -- five reshuffles of the batches into two halves, ten kernel ridge models averaged. The port had replaced it with a single fixed two-way split for ADT pairs and leave-one-batch-out otherwise, so the result depended on the order the batches happened to come in. `--n_repeats` and `--seed` are now arguments; the unused `--distance_method` and `--n_pcs` are gone (PR #32).

* `guanlab_dengkw_pm`: Only map the predictions back through the mod2 SVD when that SVD was actually fitted. When the target had fewer features than `n_mod2`, the method raised `NameError: embedder_mod2`. Unsupported modality pairs now exit 99 (non-applicable) instead of raising a `KeyError` (PR #32).

* `novel_train`: Seed the fallback that picks the internal validation batches, and log which ones were picked. Both `bmmc_multiome` datasets take this path, so the checkpoint `novel` kept differed from run to run (PR #46).

* `process_dataset`: Fall back to holding out a quarter of the batches when the dataset has no `obs["is_train"]`, rather than silently producing four empty h5ads. `obs["is_train"]` carries the NeurIPS 2021 competition split and stays optional; `obs["cell_type"]` is now declared and required (PR #28).

* Fix the component paths, build paths and `rename_keys` separator in the helper scripts, which prevented `scripts/create_datasets/test_resources.sh` and both `run_test.sh` scripts from running at all (PR #22).

* `cellmapper_scvi`: Fix postfix due to breaking changes in package (PR #23).

* `simple_mlp_predict`: Size the model output with `input_train_mod2.n_vars` instead of `input_test_mod1.n_vars`. The two only coincide on the test resources, so this crashed on every real dataset (PR #24).

* `lm`: Add an intercept column to the design matrix. `fastLm()` uses the matrix as is, so the model was forced through the origin and could never fit the mean expression level. Improves 28 of the 32 metric/dataset combinations on the test resources (PR #25).

* `mse`: Coerce both layers to sparse before differencing them. A method returning a dense `normalized` layer made the metric crash with `AttributeError: 'matrix' object has no attribute 'power'` instead of producing a score (PR #26).

* `process_dataset`: Add the `--seed` argument the API and README already advertised, and pass it through from the `process_datasets` workflow. Without it `par$seed` was `NULL`, and `set.seed(NULL)` re-seeds from the clock -- so the test-cell and ATAC-peak subsampling were not reproducible (PR #27).

* `novel`: Record every all-zero training feature in `uns["removed_vars"]`, not just the first. With more than one such feature, `novel_predict` left extra columns in the test matrix and the model dimensions no longer lined up (PR #33).

* `correlation`: Score `overall_pearson` and `overall_spearman` as 0 when either matrix is constant, matching what the per-cell and per-gene metrics already do. The `zeros` control returned `NA` for both, so the negative end of the scale was missing for two of the six metrics (PR #34).

* `process_dataset`: Sample ATAC peaks by position and cap the sample at the number of non-zero peaks. The guard tested `ncol(ad2) > 10000` but sampled from the non-zero peaks only, so a dataset with many peaks but fewer than 10000 non-zero ones errored out (PR #35).

* `novel`: Merge `wf_method.yaml` rather than `comp_method.yaml`, matching `simple_mlp`. It is a Nextflow-only workflow, so the executable-based output check never applied to it (PR #44).

* `simple_mlp`: Drop the `input_transform` key from the `simple_mlp_predict` call, which is not an argument of that component (PR #44).

* `cellmapper_scvi`: Bump the base image from `openproblems/base_pytorch_nvidia:1.0.0` to `:1`. (PR #53).

* `novel_predict`: Move the model and the input batch onto the selected device. It picked `cuda:0` when a GPU was present but never used it, so the component requested a GPU node and ran inference on CPU (PR #54).

* `correlation`, `mse`: Score non-finite predictions as zero instead of halting. `sd()` on a prediction holding `NaN` returns `NA`, so `correlation` died on `if (NA)` rather than scoring the method, and `mse` wrote a `NaN` score. A method emitting `NaN` should land at the bottom of the scale, not take the metric down with it (PR #59).

* `babel_train`, `scbutterfly`: Exit 99 for the modality pairs they do not support, rather than raising a `ValueError`. Both are GEX<->ATAC only, so on the four CITE datasets they failed with exit 1 and were retried three times each (PR #59).

* `novel_predict`: Test `uns["removed_vars"]` for length rather than truthiness. It round-trips through h5ad as an array, so the check raised "The truth value of an array with more than one element is ambiguous" as soon as training dropped more than one all-zero feature (PR #59).

* `senkin_tmp_train`: Build on `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` with `tensorflow[and-cuda]`, and ask for a `gpu`. TensorFlow was installed on a PyTorch base shipping CUDA 13 while TF is built against CUDA 12, so it silently trained on CPU and hit its walltime on all eight datasets. `openproblems/base_tensorflow_nvidia:1` reaches the GPU but is frozen at TF 2.17, below the 2.20 the method requires -- nvcr no longer publishes TensorFlow images, so that base wants rebuilding on a CUDA image in openproblems-bio/core (PR #59).

* `ss_opm`: Compute the per-cell statistics in `build_metadata()` one row block at a time. Densifying the whole normalized layer asked for a 222 GiB allocation on the multiome datasets, which fits on no node (PR #59).

# task_predict_modality 0.1.1

## NEW FUNCTIONALITY

* Added CellMapper method (two variants: simple PCA/CCA fallback and modality-specific scvi-tools models for joint mod1 representation) (PR #10)

* Added Novel method (PR #2).

* Added Simple MLP method (PR #3).

## MINOR CHANGES

* Bump image version for `openproblems/base_*` images to 1 -- a sliding release (PR #9).

* Bump Viash version to 0.9.4 (PR #12), and to 0.9.7 (PR #15).

## BUG FIXES

* `cellmapper_linear`, `cellmapper_scvi`: Fix NaNs in the linear variant, use the counts layer for the scvi models, disable HVG selection by default and correct the PCA key (PR #14).

* `cellmapper_linear`, `guanlab_dengkw_pm`: Minor script corrections (PR #15).

# task_predict_modality 0.1.0

Initial release after migrating the codebase.

## NEW FUNCTIONALITY

* Control methods: Solution, Mean per gene, Random Predictions, Zeros.

* Methods: Guanlab-dengkw, KNNR, Linear Model

* Metrics: MAE, Mean pearson / spearman, RMSE

## MAJOR CHANGES

* Refactored the API schema.
