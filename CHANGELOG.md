# task_predict_modality 0.2.0

## BUG FIXES

* `guanlab_dengkw_pm`: Only map the predictions back through the mod2 SVD when that SVD was actually fitted. When the target had fewer features than `n_mod2`, the method raised `NameError: embedder_mod2`. Unsupported modality pairs now exit 99 (non-applicable) instead of raising a `KeyError` (PR #32).

## NEW FUNCTIONALITY

* Added `src/utils/exit_codes.py`, so components can mark themselves non-applicable for a dataset (PR #32).

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
