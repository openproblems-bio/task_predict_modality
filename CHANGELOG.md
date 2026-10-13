# task_predict_modality 0.2.0

Adds eight methods, selects the ATAC target peaks by `hvg_score`, and fixes the bugs surfaced by the first full benchmark runs.

## NEW FUNCTIONALITY

* `babel`: Added BABEL, a cross-modal autoencoder with a chromosome-split ATAC encoder. Only ATAC -> GEX is exposed: GEX -> ATAC collapsed to predicting each peak's base rate regardless of the input (PR #19, #58, #59, #66).

* `cellmapper_linear`, `cellmapper_scvi`: Added CellMapper, which maps the target modality from the training to the test cells by k-NN in a PCA/CCA space or in a modality-specific scvi-tools latent space (PR #10, #14, #15, #23, #38, #53, #58, #59, #74).

* `novel`: Added Novel, an encoder-decoder MLP on the LSI of the GEX/ATAC input or the raw ADT input, from the NeurIPS 2021 competition (PR #2, #33, #37, #41, #44, #46, #47, #54, #57, #58, #59, #76).

* `scbutterfly`: Added scButterfly-B, a dual VAE with an adversarial translator, for GEX <-> ATAC. Its feature selection only applies to the input modality, so every target feature is modelled, and every predicted cell gets the level and spread of the target layer from `src/utils/cell_scale.py` (PR #20, #59, #64, #71, #81).

* `scipenn`: Added sciPENN, a recurrent network with skip connections, for GEX -> ADT (PR #72, #77).

* `senkin_tmp`: Added the LightGBM + bidirectional GRU ensemble of team senkin & tmp, the best CITE-seq submission of the NeurIPS 2022 competition, for GEX -> ADT. LightGBM trains at learning rate 0.1 for at most 100 rounds rather than at 0.01 for up to 10000, which took days per dataset, and every predicted cell gets its protein level and spread back from `src/utils/cell_scale.py` (PR #18, #59, #68, #70, #73, #75, #80).

* `simple_mlp`: Added the MLP ensemble of team AXX from the NeurIPS 2021 competition (PR #3, #24, #37, #44, #57, #58, #59).

* `ss_opm`: Added the winning solution of the NeurIPS 2022 competition, an encoder-decoder MLP on SVD-reduced inputs and targets. The inputs the original derived from the competition tables (per-cell and per-batch statistics, the HGNC/Reactome CITE gene masks) are rebuilt from the task's files, and every predicted cell gets its target level and spread back from `src/utils/cell_scale.py` (PR #16, #58, #59, #69, #80).

* Added `src/utils/exit_codes.py`, so components can mark themselves non-applicable for a dataset (PR #32).

* Added `src/utils/cell_scale.py`, which predicts each cell's target level and spread from its input, for methods that do not predict on the scale of the target layer (PR #80).

* `run_benchmark`: Run parameterised methods once per named paramset from `info.variants` or the new `--paramsets` file, tag scores with `paramset_name`/`paramset`, and allow `--methods_include`/`--methods_exclude` to target `<method_id>.<paramset_name>`. The `info.variants` of `cellmapper_linear` and `cellmapper_scvi` are run by default (ported from openproblems-bio/task_template#23, PR #74).

* Added `scripts/run_benchmark/run_full_denbi.sh` to run the full benchmark on de.NBI (PR #21).

## MAJOR CHANGES

* `process_dataset`: When the target modality is ATAC, keep the 10k peaks with the highest `hvg_score` instead of 10k random peaks, so the target is the part of the chromatin accessibility that actually differs between cells (PR #78).

* `process_dataset`: Remove cells without counts in either modality, after the peak selection. Empty cells have no profile to predict, and made `cellmapper_scvi` fail on `pbmc_cite` (43 cells without ADT counts) (PR #78).

* `run_benchmark`: Replace `--method_ids` with `--methods_include`/`--methods_exclude`, and add `--metrics_include`/`--metrics_exclude` (ported from openproblems-bio/task_template#20, PR #74).

## MINOR CHANGES

* `comp_method`: Run `check_config.py` as part of the component tests, so method metadata is validated like control methods and metrics already are (PR #37).

* `correlation`: Read the paired correlations off proxyC's sparse diagonal instead of `diag(dynutils::calculate_similarity(...))`, which densified an `n_features^2` matrix first. Scores are unchanged (PR #36).

* `correlation`: `overall_pearson` and `overall_spearman` are a single correlation of the flattened matrices, not a mean of correlations -- the descriptions said the latter. Also spelled out the zero-variance convention on all six metrics (PR #43).

* `file_train_mod1`, `file_train_mod2`, `file_test_mod1`, `file_test_mod2`: Declare `uns["modality"]`, which `process_dataset` already writes and which six methods and the `run_benchmark` workflow already read (PR #29).

* `file_test_mod2`: Declare `uns["normalization_id"]`, which `run_benchmark` reads off this file to decide which method to run on which dataset (PR #30).

* `file_test_mod2`: Label this file "Solution" and say in the description that only the metrics and control methods receive it. It holds the ground truth, but read like just another input (PR #49).

* `knnr_py`, `knnr_r`, `lm`, `guanlab_dengkw_pm`: Move `documentation_url` and `repository_url` out of `info` and into the top-level `links` (PR #37).

* `knnr_py`, `knnr_r`: Ask for `highmem` rather than `midmem`, and `lowcpu` rather than `midcpu` for `knnr_r`. `knnr_py` peaks at 82 GB against a 50 GB request, and was OOM-killed on the larger datasets (PR #58).

* `lm`: Drop the unused `n_cores` and ask for `lowcpu` rather than `highcpu`. The per-gene loop is `pbapply::pblapply()` without a cluster, so it has always run on one core (PR #42).

* `lm`: Fit every column of mod2 in one `solve()` of the normal equations instead of calling `RcppArmadillo::fastLm()` once per column. Every column shares the same design matrix, so the old loop redid an `n x n_pcs` decomposition for each of the ~229k ATAC peaks. Predictions are unchanged; `RcppArmadillo` and `pbapply` are no longer needed (PR #58).

* `mse`: Write the unbounded maximum as `"+.inf"` rather than `"+inf"`, which is the literal the metric schema accepts (PR #31).

* `run_benchmark`: Write the commit the workflow ran from and the launch time into `task_info.yaml`, instead of publishing `_viash.yaml` verbatim (ported from openproblems-bio/task_template#18, PR #74).

* `run_benchmark`: Emit one dataset metadata entry per dataset by de-duplicating on `dataset_id`, rather than by keeping only the `log_cp10k` states. The old filter emitted nothing at all if a dataset ever arrived under a different normalization (PR #48).

* `solution`, `zeros`: Ask for `lowmem` rather than `midmem` -- they use 0.5 GB and 11 GB of the 50 GB they asked for (PR #58).

* Point the `## VIASH START` blocks at files that exist. Several still referenced the openproblems-v2 monorepo layout or the pre-`normal/`-`swap/` resource layout, so running a script directly for debugging failed on the first read. Also added the missing `meta` to `knnr_r`'s block and replaced the borrowed `--id cxg_mouse_pancreas_atlas` in `run_test_local.sh` (PR #45).

* Bump image version for `openproblems/base_*` images to 1 -- a sliding release (PR #9).

* Bump Viash version to 0.9.4 (PR #12), and to 0.9.7 (PR #15).

* Added Benjamin Frey and Vladimir Shitov as authors (PR #65).

## BUG FIXES

* `correlation`: Score `overall_pearson` and `overall_spearman` as 0 when either matrix is constant, matching what the per-cell and per-gene metrics already do. The `zeros` control returned `NA` for both, so the negative end of the scale was missing for two of the six metrics (PR #34).

* `correlation`, `mse`: Score non-finite predictions as zero instead of halting. `sd()` on a prediction holding `NaN` returns `NA`, so `correlation` died on `if (NA)` rather than scoring the method, and `mse` wrote a `NaN` score. A method emitting `NaN` should land at the bottom of the scale, not take the metric down with it (PR #59).

* `guanlab_dengkw_pm`: Restore the consensus scheme of the original submission -- five reshuffles of the batches into two halves, ten kernel ridge models averaged. The port had replaced it with a single fixed two-way split for ADT pairs and leave-one-batch-out otherwise, so the result depended on the order the batches happened to come in. `--n_repeats` and `--seed` are now arguments; the unused `--distance_method` and `--n_pcs` are gone (PR #32).

* `guanlab_dengkw_pm`: Only map the predictions back through the mod2 SVD when that SVD was actually fitted. When the target had fewer features than `n_mod2`, the method raised `NameError: embedder_mod2`. Unsupported modality pairs now exit 99 (non-applicable) instead of raising a `KeyError` (PR #32).

* `guanlab_dengkw_pm`: Limit the BLAS thread pool to `meta["cpus"]`. Nothing constrains cores on the cluster, so kernel ridge sized its pool from the node's 64 cores against a 30-core allocation (PR #59).

* `guanlab_dengkw_pm`: Solve the kernel ridge regression with `scipy.linalg.cho_factor()`/`cho_solve()` instead of `KernelRidge`, whose `scipy.linalg.solve()` raised a `MemoryError` or segfaulted beyond ~30k cells with the OpenBLAS in this image. Predictions are unchanged; the method failed on every dataset except `bmmc_multiome` (PR #79).

* `guanlab_dengkw_pm`: Allow anndata to write nullable strings, which newer anndata versions refuse by default (PR #15).

* `lm`: Add an intercept column to the design matrix. `fastLm()` uses the matrix as is, so the model was forced through the origin and could never fit the mean expression level. Improves 28 of the 32 metric/dataset combinations on the test resources (PR #25).

* `mse`: Coerce both layers to sparse before differencing them. A method returning a dense `normalized` layer made the metric crash with `AttributeError: 'matrix' object has no attribute 'power'` instead of producing a score (PR #26).

* `process_dataset`: Fall back to holding out a quarter of the batches when the dataset has no `obs["is_train"]`, rather than silently producing four empty h5ads. `obs["is_train"]` carries the NeurIPS 2021 competition split and stays optional; `obs["cell_type"]` is now declared and required (PR #28).

* `process_dataset`: Add the `--seed` argument the API and README already advertised, and pass it through from the `process_datasets` workflow. Without it `par$seed` was `NULL`, and `set.seed(NULL)` re-seeds from the clock -- so the test-cell and ATAC-peak subsampling were not reproducible (PR #27).

* Fix the component paths, build paths and `rename_keys` separator in the helper scripts, which prevented `scripts/create_datasets/test_resources.sh` and both `run_test.sh` scripts from running at all (PR #22).

# task_predict_modality 0.1.0

Initial release after migrating the codebase.

## NEW FUNCTIONALITY

* Control methods: Solution, Mean per gene, Random Predictions, Zeros.

* Methods: Guanlab-dengkw, KNNR, Linear Model

* Metrics: MAE, Mean pearson / spearman, RMSE

## MAJOR CHANGES

* Refactored the API schema.
