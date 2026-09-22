"""Shared scButterfly setup used by both the train and predict components.

scButterfly has no transform-only preprocessing API — ``data_preprocessing`` refits
HVG/peak-filter/TF-IDF on whatever data it is given. To run inference faithfully in a
separate process, predict must rebuild the *same* paired object and re-run the *same*
deterministic preprocessing as train, then load the saved weights. This module holds
that shared construction so train and predict stay in lock-step.
"""

import contextlib
import logging

from exit_codes import exit_non_applicable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse

import chrom_utils

logger = logging.getLogger(__name__)


def apply_runtime_patches():
    """Patch scButterfly/torch quirks. Call before importing scButterfly.

    - legacy ``size_average``/``reduce`` loss kwargs -> ``reduction``
    - BCE on a float32 sigmoid can drift >1.0 -> clamp input to [0,1]
    - scButterfly hardcodes ``.cuda()``; make it a no-op when no GPU is present
    - ``TFIDF`` builds dense (n_peaks, n_cells) tiles -> sparse equivalent
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _patch_legacy_reduction(cls):
        orig_init = cls.__init__

        def _init(self, *args, size_average=None, reduce=None, reduction="mean", **kwargs):
            if size_average is not None or reduce is not None:
                reduction = "mean" if (size_average in (None, True)) else "sum"
            orig_init(self, *args, reduction=reduction, **kwargs)

        cls.__init__ = _init

    for _loss_cls in (nn.MSELoss, nn.BCELoss, nn.L1Loss):
        _patch_legacy_reduction(_loss_cls)

    _orig_bce = F.binary_cross_entropy

    def _safe_bce(input, target, *args, **kwargs):
        input = torch.nan_to_num(input, nan=0.0).clamp(0.0, 1.0)
        return _orig_bce(input, target, *args, **kwargs)

    F.binary_cross_entropy = _safe_bce

    # Log the device up front. `torch.version.cuda` is None for a CPU-only wheel
    # and e.g. "11.7" for the cu117 one, so the two failure modes — wrong wheel
    # installed vs. right wheel but no device visible — are distinguishable in the
    # run log instead of only showing up as an unexplained slow run.
    logger.info("torch %s (CUDA build: %s)", torch.__version__, torch.version.cuda)
    if torch.cuda.is_available():
        logger.info(
            "CUDA available: %d device(s), using %s",
            torch.cuda.device_count(), torch.cuda.get_device_name(0),
        )
    else:
        logger.warning("CUDA not available — running scButterfly on CPU (slow).")
        torch.Tensor.cuda = lambda self, *a, **k: self
        nn.Module.cuda = lambda self, *a, **k: self

    # Must come after the torch patches above: importing scButterfly.data_processing
    # pulls in the whole package, including the modules those patches target.
    from scButterfly import data_processing

    data_processing.TFIDF = _sparse_tfidf
    logger.info("Patched scButterfly TFIDF with the sparse implementation.")


def _sparse_tfidf(count_mat):
    """Sparse, O(nnz) drop-in for ``scButterfly.data_processing.TFIDF``.

    The upstream version ``np.tile``s the per-cell and per-peak totals into dense
    ``(n_peaks, n_cells)`` matrices and divides/multiplies densely, so it costs
    ``O(n_peaks * n_cells)`` no matter how sparse the input is — 37 GiB per array at
    NeurIPS2021 Multiome scale, with three or four alive at once. That is what puts
    the component at ~265 GB peak, and what OOMs it on the 2022 Multiome datasets.

    Same arithmetic in the same order, evaluated on the CSR ``.data`` array::

        out[c, p] = X[c, p] / cell_total[c] * log(1 + n_cells / peak_total[p])

    scButterfly binarizes before calling this, so both totals are exact integers in
    float32 and summation order cannot matter: the result is bit-identical to the
    dense one. Entries stay strictly positive (``peak_total <= n_cells`` keeps the
    log argument above 1), so the sparsity pattern is preserved too.

    Degenerate rows differ, in the safe direction: a cell with no stored entries
    used to come back all-NaN from ``0 / 0`` and now stays zero.

    The second and third return values are the *untiled* per-cell and per-peak
    vectors instead of the dense tiles. They exist only for ``inverse_TFIDF``, which
    neither this component nor scButterfly's own training code ever calls.
    """
    X = count_mat.tocsr() if issparse(count_mat) else csr_matrix(count_mat)
    n_cells = X.shape[0]

    cell_totals = np.asarray(X.sum(axis=1)).ravel()
    peak_totals = np.asarray(X.sum(axis=0)).ravel()
    idf = np.log(1 + 1.0 * n_cells / peak_totals)

    # Repeat each cell's total across the entries stored for that cell so the
    # expression below lines up term for term with the dense one.
    per_entry_cell_total = np.repeat(cell_totals, np.diff(X.indptr))
    data = X.data / per_entry_cell_total * idf[X.indices]

    out = csr_matrix((data, X.indices, X.indptr), shape=X.shape)
    return out, cell_totals, idf


@contextlib.contextmanager
def suppress_unused_postprocessing():
    """Make ``sc.pp.pca``/``sc.pp.neighbors`` no-ops for the duration of the block.

    ``Model.test`` runs both, on *both* predicted matrices, whenever it is not asked
    to draw figures — and there is no flag to turn it off. Nothing reads the results
    back: :func:`extract_predictions` only touches ``.X`` and ``.var_names``. On the
    ATAC side it is a PCA over a near-dense ``n_test x n_peaks`` matrix, which
    dominates the predict step.
    """
    orig_pca, orig_neighbors = sc.pp.pca, sc.pp.neighbors
    sc.pp.pca = lambda *args, **kwargs: None
    sc.pp.neighbors = lambda *args, **kwargs: None
    try:
        yield
    finally:
        sc.pp.pca, sc.pp.neighbors = orig_pca, orig_neighbors


def detect_direction(train_mod1, train_mod2):
    """Return ('GEX2ATAC'|'ATAC2GEX', mod1, mod2), exiting 99 on non-Multiome data."""
    mod1 = train_mod1.uns["modality"]
    mod2 = train_mod2.uns["modality"]
    if {mod1, mod2} != {"GEX", "ATAC"}:
        exit_non_applicable(
            f"scbutterfly only supports Multiome GEX<->ATAC, got mod1={mod1}, mod2={mod2}"
        )
    direction = "GEX2ATAC" if mod2 == "ATAC" else "ATAC2GEX"
    return direction, mod1, mod2


def _to_counts_X(adata):
    """AnnData whose .X is the raw counts layer as float32, carrying obs and var.

    Counts are stored as float64; scButterfly's model is float32 and its data loader
    does not cast, so feed float32 to avoid a dtype mismatch.

    Everything else is dropped rather than copied. ``sc.concat`` below discards the
    other layers, ``obsm['gene_activity']`` and ``uns`` anyway — the placeholder
    block has none of them and anndata intersects — so a full ``.copy()`` only
    duplicates the largest matrices in the file to throw them away a moment later.
    """
    return ad.AnnData(
        X=adata.layers["counts"].astype(np.float32),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )


def _placeholder_block(template_adata, n_rows, obs_names):
    """Placeholder rows for the target modality's test cells.

    These are the modality being predicted (no ground truth) and are never used as
    model input — they exist only to satisfy scButterfly's paired same-cells layout.
    Fill by tiling real training rows (not zeros): all-zero rows/cols make per-cell
    normalization and TF-IDF divide by zero, producing NaNs that break BCE.
    """
    src = template_adata.X
    idx = np.arange(n_rows) % max(1, src.shape[0])
    # Gather the rows on the CSR directly. Densifying the training block first cost
    # n_train * n_features floats (>12 GiB of ATAC at Multiome scale) and bought
    # nothing — the gathered rows hold the same values either way.
    if issparse(src):
        X = src.tocsr()[idx].astype(np.float32)
    else:
        X = csr_matrix(np.asarray(src)[idx].astype(np.float32))
    obs = pd.DataFrame(index=obs_names)
    return ad.AnnData(X=X, obs=obs, var=template_adata.var.copy())


def _concat_blocks(train_block, test_block):
    """Stack the train and test blocks, keeping the training var order.

    Both blocks are built from the same ``var``, so the outer join already comes back
    in the training order and reindexing is a no-op whose ``.copy()`` would duplicate
    the whole matrix. The explicit reindex stays as a fallback if that stops holding.
    """
    out = sc.concat([train_block, test_block], axis=0, join="outer")
    if not out.var_names.equals(train_block.var_names):
        return out[:, train_block.var_names].copy()
    # Materialising the reindexed view also dropped categories left unused by the
    # concat (obs['batch'] keeps all donors otherwise). Nothing downstream reads
    # obs, but do it anyway so the result matches the previous code exactly.
    for frame in (out.obs, out.var):
        for col in frame.columns:
            if isinstance(frame[col].dtype, pd.CategoricalDtype):
                frame[col] = frame[col].cat.remove_unused_categories()
    return out


def build_butterfly(train_mod1, train_mod2, test_mod1, n_top_genes, Butterfly):
    """Build + preprocess + construct a Butterfly for the given data.

    Returns a dict with the constructed ``butterfly``, the resolved ``direction``,
    ``test_id`` (indices of the test cells), ``chrom_list``, and the preprocessed
    target-modality var names. Deterministic given the same inputs, so train and
    predict produce identical architecture/preprocessing.
    """
    direction, mod1, mod2 = detect_direction(train_mod1, train_mod2)
    logger.info("Direction: %s (mod1=%s, mod2=%s)", direction, mod1, mod2)

    # Assign RNA/ATAC roles: scButterfly wants RNA_data=GEX, ATAC_data=ATAC.
    if mod1 == "GEX":
        rna_train, atac_train = train_mod1, train_mod2
        rna_test, atac_test = test_mod1, None
    else:
        atac_train, rna_train = train_mod1, train_mod2
        atac_test, rna_test = test_mod1, None

    n_train = train_mod1.n_obs
    n_test = test_mod1.n_obs
    test_obs_names = [f"test_{i}" for i in range(n_test)]

    rna_train_c = _to_counts_X(rna_train)
    atac_train_c = _to_counts_X(atac_train)
    rna_train_c.obs_names = [f"train_{i}" for i in range(n_train)]
    atac_train_c.obs_names = [f"train_{i}" for i in range(n_train)]

    if rna_test is not None:
        rna_test_c = _to_counts_X(rna_test)
        rna_test_c.obs_names = test_obs_names
    else:
        rna_test_c = _placeholder_block(rna_train_c, n_test, test_obs_names)

    if atac_test is not None:
        atac_test_c = _to_counts_X(atac_test)
        atac_test_c.obs_names = test_obs_names
    else:
        atac_test_c = _placeholder_block(atac_train_c, n_test, test_obs_names)

    RNA_data = _concat_blocks(rna_train_c, rna_test_c)
    ATAC_data = _concat_blocks(atac_train_c, atac_test_c)

    train_id = list(range(n_train))
    test_id = list(range(n_train, n_train + n_test))

    # Deterministic ~10% validation carve (not five_fold_split_dataset).
    rng = np.random.RandomState(0)
    shuffled = list(train_id)
    rng.shuffle(shuffled)
    n_val = max(1, int(0.1 * n_train))
    validation_id = sorted(shuffled[:n_val])
    train_id_final = sorted(shuffled[n_val:])

    # Chromosome ordering (peaks contiguous per chrom + chrom_list).
    sort_index, chrom_list = chrom_utils.sorted_chrom_order(ATAC_data)
    ATAC_data = chrom_utils.apply_sort(ATAC_data, sort_index)

    butterfly = Butterfly()
    butterfly.load_data(RNA_data, ATAC_data, train_id_final, test_id, validation_id)
    butterfly.data_preprocessing(n_top_genes=n_top_genes)
    butterfly.augmentation(aug_type=None)

    # scButterfly casts to float32 only on the CUDA path; force float32 for CPU.
    for _attr in ("RNA_data_p", "ATAC_data_p"):
        _adata = getattr(butterfly, _attr, None)
        if _adata is not None and _adata.X is not None:
            _adata.X = _adata.X.astype(np.float32)

    # Rebuild chrom_list from the preprocessed (peak-filtered) ATAC so model dims match.
    atac_p = getattr(butterfly, "ATAC_data_p", None)
    if atac_p is not None and atac_p.n_vars != sum(chrom_list):
        if "chrom" not in atac_p.var.columns:
            atac_p.var["chrom"] = [chrom_utils.parse_chrom(v) for v in atac_p.var_names]
        chrom_list = chrom_utils.chrom_counts(atac_p)
    logger.info("chrom_list: %d chromosomes, %d peaks", len(chrom_list), sum(chrom_list))

    butterfly.construct_model(chrom_list=chrom_list)

    # var names of the preprocessed target modality, used to map predictions back.
    target_p = butterfly.RNA_data_p if direction == "ATAC2GEX" else butterfly.ATAC_data_p

    return {
        "butterfly": butterfly,
        "direction": direction,
        "n_train": n_train,
        "n_test": n_test,
        "test_id": test_id,
        "chrom_list": chrom_list,
        "target_p_var_names": list(target_p.var_names),
    }


def extract_predictions(built, A2R_predict, R2A_predict, target_var_names):
    """Select the target-modality prediction, name its vars, and scatter to target order."""
    direction = built["direction"]
    n_train, n_test = built["n_train"], built["n_test"]
    pred = A2R_predict if direction == "ATAC2GEX" else R2A_predict

    # tensor2adata gives integer var names; assign the preprocessed target var names.
    p_names = built["target_p_var_names"]
    if pred.n_vars == len(p_names):
        pred.var_names = p_names
    else:
        logger.warning("Prediction vars (%d) != preprocessed target vars (%d)",
                       pred.n_vars, len(p_names))

    if pred.n_obs == (n_train + n_test):
        pred = pred[built["test_id"]].copy()
    elif pred.n_obs != n_test:
        logger.warning("Unexpected prediction rows: %d (n_test=%d)", pred.n_obs, n_test)

    out = chrom_utils.scatter_to_target(pred, target_var_names)
    overlap = len(set(pred.var_names) & set(target_var_names))
    logger.info("Var overlap with target: %d / %d", overlap, len(target_var_names))
    return out
