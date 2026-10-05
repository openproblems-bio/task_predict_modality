"""Map scButterfly's predictions onto the scale of the target's ``normalized`` layer.

scButterfly predicts its own preprocessing of the target modality, which is not the
layer the benchmark scores:

- GEX: ``log1p`` of each cell's counts scaled to a total of ``target_sum``, the median
  count total of the cells (scanpy's ``normalize_total`` default), where the dataset
  has log CP10k. With every gene modelled, the two differ by a constant factor inside
  the log, so :func:`to_log_cp10k` converts exactly.
- ATAC: binarized peaks, TF-IDF-transformed and divided by the global maximum,
  predicted through a sigmoid, where the dataset has its own TF-IDF (0 to ~20). There is
  no closed form, so :class:`PeakCalibration` fits one least-squares line per peak on
  cells held out from training.

Both maps are increasing, so per-gene and per-peak rankings stay as the model made them
(a peak whose fitted slope is zero becomes constant, see below); what they fix is the
level and spread that the per-cell profiles, overall correlations and RMSE/MAE score.
"""

import numpy as np
import scipy.sparse as sp

LOG_CP10K_TOTAL = 1e4


def to_log_cp10k(predictions, target_sum):
    """Convert ``log1p(counts / total * target_sum)`` to ``log1p(counts / total * 1e4)``.

    The RNA decoder ends in a LeakyReLU, so small negative outputs (no expression) are
    clipped to zero first.
    """
    counts_per_target_sum = np.expm1(np.clip(predictions, 0, None))
    return np.log1p(counts_per_target_sum * (LOG_CP10K_TOTAL / target_sum)).astype(np.float32)


class PeakCalibration:
    """Per-peak least-squares line from scButterfly's ATAC output to the target layer.

    A peak whose held-out predictions do not correlate positively with the truth gets
    slope zero, i.e. its mean: a negative slope would turn the model's ranking of the
    cells around, which it is not trained to do.
    """

    def fit(self, predictions, targets):
        """Fit on ``predictions`` (cells x peaks, dense) and ``targets`` (dense or sparse)."""
        predictions = np.asarray(predictions, dtype=np.float64)
        prediction_mean = predictions.mean(axis=0)
        centered = predictions - prediction_mean
        if sp.issparse(targets):
            target_mean = np.asarray(targets.mean(axis=0)).ravel()
            covariance = np.asarray(sp.csr_matrix(targets).multiply(centered).sum(axis=0)).ravel()
        else:
            targets = np.asarray(targets, dtype=np.float64)
            target_mean = targets.mean(axis=0)
            covariance = (targets * centered).sum(axis=0)
        variance = (centered**2).sum(axis=0)
        self.slope_ = np.where(variance > 0, np.maximum(covariance, 0) / np.where(variance > 0, variance, 1), 0)
        self.intercept_ = target_mean - self.slope_ * prediction_mean
        return self

    def apply(self, predictions):
        return (np.asarray(predictions) * self.slope_ + self.intercept_).astype(np.float32)
