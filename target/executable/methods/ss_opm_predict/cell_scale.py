"""Restore each cell's level and spread for predictions that only get the per-cell profile right.

Methods written for the NeurIPS 2022 Kaggle competition, which scored nothing but the per-cell Pearson correlation,
predict a z-scored profile for each cell. Mapped back with one global slope and intercept, every predicted cell ends up
with the same mean and standard deviation, although the true per-cell mean varies several-fold between cells; per-gene
correlations, overall correlations and RMSE/MAE then score that missing information, not the method.

`CellScale` learns each training cell's target mean and log standard deviation from its input profile (a ridge
regression on a truncated SVD of the normalized input plus two sequencing-depth features) and gives every standardized
prediction its own mean back, and its own standard deviation times one shrinkage factor. The factor is fitted by least
squares on predictions for training cells (out-of-fold where the method has them): where the predicted profile
correlates only weakly with the truth, the error-minimising spread is smaller than the cell's standard deviation.
Per-cell correlations are unchanged by construction.

The fitted model is a handful of arrays, saved with `numpy.savez`, so that it can be loaded in another image without
unpickling library objects.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV

RIDGE_ALPHAS = np.logspace(-2, 4, 13)
EPSILON = 1e-3  # keeps log(sd) finite for constant target vectors


def row_mean_std(matrix):
    """Mean and standard deviation of every row of a dense or sparse matrix."""
    if sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        mean = np.asarray(matrix.mean(axis=1)).ravel()
        mean_of_squares = np.asarray(matrix.multiply(matrix).mean(axis=1)).ravel()
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        mean = matrix.mean(axis=1)
        mean_of_squares = (matrix**2).mean(axis=1)
    return mean, np.sqrt(np.maximum(mean_of_squares - mean**2, 0))


def standardize_rows(matrix):
    """Z-score every row; a constant row becomes all zeros."""
    matrix = np.asarray(matrix, dtype=np.float64)
    mean, std = row_mean_std(matrix)
    return (matrix - mean[:, None]) / np.where(std > 0, std, 1)[:, None]


def depth_features(counts):
    """log1p of the total and of the number of detected features per cell."""
    counts = sp.csr_matrix(counts)
    total = np.asarray(counts.sum(axis=1)).ravel()
    detected = np.diff(counts.indptr)
    return np.column_stack([np.log1p(np.maximum(total, 0)), np.log1p(detected)])


class CellScale:
    """Predict each cell's target mean and standard deviation from its input, and apply them to standardized predictions.

    `inputs` is the normalized input modality (cells x input features), `counts` its raw counts (or the normalized
    values when there are none; only used for the two depth features), `targets` the normalized target modality.
    """

    def __init__(self, n_components=64, random_state=0):
        self.n_components = n_components
        self.random_state = random_state
        self.spread_ = 1.0

    def _features(self, inputs, counts):
        inputs = sp.csr_matrix(inputs, dtype=np.float32)
        projected = np.asarray(inputs @ self.components_.T)
        features = np.hstack([projected, depth_features(inputs if counts is None else counts)])
        return (features - self.feature_mean_) / self.feature_std_

    def fit(self, inputs, targets, counts=None):
        inputs = sp.csr_matrix(inputs, dtype=np.float32)
        n_components = min(self.n_components, min(inputs.shape) - 1)
        self.components_ = TruncatedSVD(n_components=n_components, random_state=self.random_state).fit(inputs).components_
        raw_features = np.hstack([np.asarray(inputs @ self.components_.T), depth_features(inputs if counts is None else counts)])
        self.feature_mean_ = raw_features.mean(axis=0)
        self.feature_std_ = np.where(raw_features.std(axis=0) > 0, raw_features.std(axis=0), 1)
        target_mean, target_std = row_mean_std(targets)
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            (raw_features - self.feature_mean_) / self.feature_std_,
            np.column_stack([target_mean, np.log(target_std + EPSILON)]),
        )
        self.coef_, self.intercept_ = ridge.coef_, ridge.intercept_
        return self

    def predict(self, inputs, counts=None):
        """Predicted target mean and standard deviation of every cell."""
        fitted = self._features(inputs, counts) @ self.coef_.T + self.intercept_
        return fitted[:, 0], np.maximum(np.exp(fitted[:, 1]) - EPSILON, 0)

    def fit_spread(self, predictions, inputs, targets, counts=None, block_size=1000):
        """Least-squares shrinkage of the predicted deviations from the cell mean, fitted on training cells."""
        mean, std = self.predict(inputs, counts)
        targets = sp.csr_matrix(targets) if sp.issparse(targets) else np.asarray(targets)
        covariance, variance = 0.0, 0.0
        for start in range(0, predictions.shape[0], block_size):
            rows = slice(start, start + block_size)
            deviations = standardize_rows(predictions[rows]) * std[rows, None]
            block_targets = targets[rows].toarray() if sp.issparse(targets) else targets[rows]
            covariance += float(((block_targets - mean[rows, None]) * deviations).sum())
            variance += float((deviations**2).sum())
        self.spread_ = covariance / variance if variance > 0 else 0.0
        return self

    def apply(self, predictions, inputs, counts=None):
        """Give each cell of `predictions` (any per-cell scale) its predicted mean and shrunken standard deviation."""
        mean, std = self.predict(inputs, counts)
        return standardize_rows(predictions) * (self.spread_ * std)[:, None] + mean[:, None]

    def save(self, path):
        np.savez(path, components=self.components_, feature_mean=self.feature_mean_, feature_std=self.feature_std_,
                 coef=self.coef_, intercept=self.intercept_, spread=self.spread_)

    @classmethod
    def load(cls, path):
        arrays = np.load(path)
        model = cls(n_components=arrays["components"].shape[0])
        model.components_, model.feature_mean_, model.feature_std_ = arrays["components"], arrays["feature_mean"], arrays["feature_std"]
        model.coef_, model.intercept_, model.spread_ = arrays["coef"], arrays["intercept"], float(arrays["spread"])
        return model
