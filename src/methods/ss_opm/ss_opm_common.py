"""Helpers shared by ss_opm_train and ss_opm_predict."""

import numpy as np
import pandas as pd
import scipy.sparse

# the cell types the original ss_opm model was trained against. only used to name the
# cell_ratio_* columns it expects; the ratios themselves are derived from the data when
# cell type labels are available.
CITE_CELL_TYPES = ["HSC", "EryP", "NeuP", "MasP", "MkP", "BP", "MoP"]

# number of batch singular-vector columns the cite model expects
N_BATCH_SV = 8


def to_sparse_csr(X):
    if scipy.sparse.issparse(X):
        return X.tocsr()
    return scipy.sparse.csr_matrix(X)


def extract_day(batch, pattern=r"d(\d+)"):
    """Pull the day out of a batch label.

    The NeurIPS 2021 batches are named `s{site}d{day}`, e.g. `s1d2`. Datasets that
    label their batches differently yield NaN, which the caller fills with 0 -- the
    model then sees a single constant day rather than failing.
    """
    return batch.astype(str).str.extract(pattern, expand=False).astype(float)


def build_metadata(
    adata,
    task_type,
    cell_type_col=None,
    group_by_batch=True,
    day_pattern=r"d(\d+)",
):
    """Build the metadata frame ss_opm expects from an AnnData.

    ss_opm was written against the Kaggle competition tables, which carry columns this
    task's API does not: `file_train_mod1.yaml` and `file_test_mod1.yaml` guarantee only
    `batch`. Everything else is either derived from `batch`, computed from the expression
    matrix, or filled with a neutral constant.

    Parameters
    ----------
    adata
        Input modality, with a `normalized` layer and `obs["batch"]`.
    task_type
        Either `"cite"` or `"multi"`; the cite model expects extra columns.
    cell_type_col
        Column in `adata.obs` holding cell type labels. When given, it drives both
        `cell_type` and the `cell_ratio_*` columns. When None -- the case for every
        dataset this task currently ships -- cell types are `"hidden"` and the ratios
        are uniform.
    group_by_batch
        Assign one group per batch. Set False to put every cell in group 0, which is
        what the predict path wants, since targets are absent and the group IDs are
        only used to look up target statistics.
    day_pattern
        Regex whose first capture group is the day within a batch label.
    """
    obs = pd.DataFrame(index=adata.obs_names)

    obs["batch"] = adata.obs["batch"].values
    obs["day"] = extract_day(adata.obs["batch"], day_pattern).fillna(0).values

    # per-cell statistics from the normalized expression layer
    X = adata.layers["normalized"]
    X_dense = X.toarray() if scipy.sparse.issparse(X) else np.asarray(X, dtype=float)

    obs["nonzero_ratio"] = (X_dense != 0).mean(axis=1)
    obs["nonzero_q25"] = np.percentile(X_dense, 25, axis=1)
    obs["nonzero_q50"] = np.percentile(X_dense, 50, axis=1)
    obs["nonzero_q75"] = np.percentile(X_dense, 75, axis=1)
    obs["mean"] = X_dense.mean(axis=1)
    obs["std"] = X_dense.std(axis=1)

    if group_by_batch:
        batches = adata.obs["batch"].unique().tolist()
        obs["group"] = adata.obs["batch"].map({b: i for i, b in enumerate(batches)}).astype(int).values
    else:
        obs["group"] = 0

    # cell type labels, when the caller can supply them
    if cell_type_col is not None and cell_type_col in adata.obs:
        obs["cell_type"] = adata.obs[cell_type_col].astype(str).values
    else:
        obs["cell_type"] = "hidden"

    # donor and technology are not in this task's file format; gender_id defaults to 0
    obs["donor"] = 0
    obs["technology"] = "unknown"

    if task_type == "cite":
        ratios = obs["cell_type"].value_counts(normalize=True)
        for cell_type in CITE_CELL_TYPES:
            obs[f"cell_ratio_{cell_type}"] = ratios.get(cell_type, 1.0 / len(CITE_CELL_TYPES))

        batch_counts = adata.obs["batch"].value_counts()
        obs["cell_count"] = adata.obs["batch"].map(batch_counts).astype(float).values

        # the originals are singular vectors of the full Kaggle batch matrix, which we
        # cannot reconstruct from a single dataset
        for i in range(N_BATCH_SV):
            obs[f"batch_sv{i}"] = 0.0

    return obs
