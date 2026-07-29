# task_predict_modality 0.2.0

## BUG FIXES

* Fix the component paths, build paths and `rename_keys` separator in the helper scripts, which prevented `scripts/create_datasets/test_resources.sh` and both `run_test.sh` scripts from running at all (PR #22).

* `cellmapper_scvi`: Fix postfix due to breaking changes in package (PR #23).

* `simple_mlp_predict`: Size the model output with `input_train_mod2.n_vars` instead of `input_test_mod1.n_vars`. The two only coincide on the test resources, so this crashed on every real dataset (PR #24).

* `mse`: Coerce both layers to sparse before differencing them. A method returning a dense `normalized` layer made the metric crash with `AttributeError: 'matrix' object has no attribute 'power'` instead of producing a score (PR #26).

* `process_dataset`: Add the `--seed` argument the API and README already advertised, and pass it through from the `process_datasets` workflow. Without it `par$seed` was `NULL`, and `set.seed(NULL)` re-seeds from the clock -- so the test-cell and ATAC-peak subsampling were not reproducible (PR #27).

# task_predict_modality 0.1.1

## NEW FUNCTIONALITY

* Added CellMapper method (two variants: simple PCA/CCA fallback and modality-specific scvi-tools models for joint mod1 representation) (PR #10)

* Added Novel method (PR #2).

* Added Simple MLP method (PR #3).

## MINOR CHANGES

* Bump image version for `openproblems/base_*` images to 1 -- a sliding release (PR #9).

* Bump Viash version to 0.9.4 (PR #12).

# task_predict_modality 0.1.0

Initial release after migrating the codebase.

## NEW FUNCTIONALITY

* Control methods: Solution, Mean per gene, Random Predictions, Zeros.

* Methods: Guanlab-dengkw, KNNR, Linear Model

* Metrics: MAE, Mean pearson / spearman, RMSE

## MAJOR CHANGES

* Refactored the API schema.
