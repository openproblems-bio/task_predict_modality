"""Shared scButterfly setup used by both the train and predict components.

scButterfly has no transform-only preprocessing API — ``data_preprocessing`` refits
HVG/peak-filter/TF-IDF on whatever data it is given. To run inference faithfully in a
separate process, predict must rebuild the *same* paired object and re-run the *same*
deterministic preprocessing as train, then load the saved weights. This module holds
that shared construction so train and predict stay in lock-step.
"""

import logging

from exit_codes import exit_non_applicable

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

import chrom_utils

logger = logging.getLogger(__name__)


def apply_runtime_patches():
    """Patch scButterfly/torch quirks. Call before importing scButterfly.

    - legacy ``size_average``/``reduce`` loss kwargs -> ``reduction``
    - BCE on a float32 sigmoid can drift >1.0 -> clamp input to [0,1]
    - scButterfly hardcodes ``.cuda()``; make it a no-op when no GPU is present
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

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — running scButterfly on CPU (slow).")
        torch.Tensor.cuda = lambda self, *a, **k: self
        nn.Module.cuda = lambda self, *a, **k: self


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
    """AnnData copy whose .X is the raw counts layer as float32.

    Counts are stored as float64; scButterfly's model is float32 and its data loader
    does not cast, so feed float32 to avoid a dtype mismatch.
    """
    out = adata.copy()
    out.X = out.layers["counts"].astype(np.float32)
    return out


def _placeholder_block(template_adata, n_rows, obs_names):
    """Placeholder rows for the target modality's test cells.

    These are the modality being predicted (no ground truth) and are never used as
    model input — they exist only to satisfy scButterfly's paired same-cells layout.
    Fill by tiling real training rows (not zeros): all-zero rows/cols make per-cell
    normalization and TF-IDF divide by zero, producing NaNs that break BCE.
    """
    import pandas as pd
    src = template_adata.X
    src = src.toarray() if hasattr(src, "toarray") else np.asarray(src)
    idx = np.arange(n_rows) % max(1, src.shape[0])
    X = csr_matrix(src[idx].astype(np.float32))
    obs = pd.DataFrame(index=obs_names)
    return ad.AnnData(X=X, obs=obs, var=template_adata.var.copy())


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

    RNA_data = sc.concat([rna_train_c, rna_test_c], axis=0, join="outer")[:, rna_train_c.var_names].copy()
    ATAC_data = sc.concat([atac_train_c, atac_test_c], axis=0, join="outer")[:, atac_train_c.var_names].copy()

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
