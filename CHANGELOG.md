# task_predict_modality 0.2.0

## MINOR CHANGES

* `novel`: Drop the first of the two identical validation passes in `train_and_valid()`. Its result was never read, so it cost one full pass over the validation set per epoch, 100 epochs per run (PR #41).

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
