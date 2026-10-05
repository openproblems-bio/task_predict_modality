"""Map scButterfly's predictions onto the scale of the target's ``normalized`` layer.

scButterfly predicts its own preprocessing of the target modality, which is not the
layer the benchmark scores:

- GEX: ``log1p`` of each cell's counts scaled to the median count total of the cells
  (scanpy's ``normalize_total`` default) instead of 1e4.
- ATAC: binarized peaks, TF-IDF-transformed and divided by the global maximum,
  predicted through a sigmoid (0 to 1), where the dataset has its own TF-IDF (0 to ~20).

:class:`ScaleCalibration` maps them with one least-squares line, fitted on training
cells scButterfly held out for validation. On the NeurIPS 2021 and 2022 Multiome
datasets it scored as well as or better than the alternatives on every metric: one line
per gene or peak (noisier, and it reorders each cell's profile), and for GEX the exact
conversion ``log1p(expm1(p) * 1e4 / target_sum)``. The exact conversion is exact for a
value, not for the model's prediction of one: an MSE-trained decoder predicts a
conditional mean in its own log space, and the concave conversion inflates it for
sparsely expressed genes (RMSE 0.553 against 0.518 for the line on ``bmmc_multiome``).

The line is increasing, so every correlation is exactly the model's; it fixes the level
and spread that RMSE scores.
"""

import numpy as np
import scipy.sparse as sp


class ScaleCalibration:
    """One least-squares line from scButterfly's output to the target layer."""

    def fit(self, predictions, targets, block_size=1000):
        """Fit on ``predictions`` (cells x features, dense) and ``targets`` (dense or sparse).

        If the held-out cells show no positive association, which only an essentially
        untrained model produces (e.g. two epochs on the test resources), the line
        matches the target's mean and standard deviation instead, so the predictions
        are rescaled rather than replaced by a constant.
        """
        n_entries = predictions.size
        prediction_sum, prediction_square_sum = 0.0, 0.0
        for start in range(0, predictions.shape[0], block_size):
            block = predictions[start:start + block_size].astype(np.float64)
            prediction_sum += block.sum()
            prediction_square_sum += (block**2).sum()
        targets = sp.csr_matrix(targets, dtype=np.float64)
        prediction_mean = prediction_sum / n_entries
        target_mean = targets.sum() / n_entries
        covariance = targets.multiply(predictions).sum() / n_entries - prediction_mean * target_mean
        variance = prediction_square_sum / n_entries - prediction_mean**2
        target_variance = (targets.data**2).sum() / n_entries - target_mean**2
        if variance <= 0:
            self.slope_ = 0.0
        elif covariance > 0:
            self.slope_ = covariance / variance
        else:
            self.slope_ = np.sqrt(max(target_variance, 0.0) / variance)
        self.intercept_ = target_mean - self.slope_ * prediction_mean
        return self

    def apply(self, predictions):
        return (np.asarray(predictions) * self.slope_ + self.intercept_).astype(np.float32)
