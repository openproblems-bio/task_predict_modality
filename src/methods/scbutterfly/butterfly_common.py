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
import torch
from scipy.sparse import csr_matrix, issparse

import chrom_utils

logger = logging.getLogger(__name__)


def apply_runtime_patches():
    """Patch scButterfly/torch quirks. Call before importing scButterfly.

    - legacy ``size_average``/``reduce`` loss kwargs -> ``reduction``
    - BCE on a float32 sigmoid can drift >1.0 -> clamp input to [0,1]
    - scButterfly hardcodes ``.cuda()``; make it a no-op when no GPU is present
    - ``TFIDF`` builds dense (n_peaks, n_cells) tiles -> sparse equivalent
    - ``Model`` densifies both whole modalities -> keep them sparse
    - per-cell ``DataLoader`` batches -> batches densified on the device
    """
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

    from scButterfly import train_model

    train_model.Model.__init__ = _keep_data_sparse(train_model.Model.__init__)
    train_model.DataLoader = _SparseBatchLoader


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


def _keep_data_sparse(original_init):
    """Wrap ``Model.__init__`` so that it keeps the data as float32 CSR matrices.

    Upstream stores ``RNA_data.X.toarray()`` and ``ATAC_data.X.toarray()``: every cell
    of both modalities, training and test, as dense float32. At NeurIPS 2022 Multiome
    scale (131k cells x 175k peaks after peak filtering) the ATAC matrix alone is 92 GB.
    The model reads only their ``.shape`` and, through its datasets, batches of rows,
    which :class:`_SparseBatchLoader` serves from the sparse matrices.
    """

    def init(self, RNA_data, ATAC_data, *args, **kwargs):
        # The networks are sized from the dim lists, not the data, so build them on
        # zero cells and attach the real matrices afterwards.
        original_init(self, RNA_data[:0], ATAC_data[:0], *args, **kwargs)
        self.RNA_data_obs, self.ATAC_data_obs = RNA_data.obs, ATAC_data.obs
        # Upstream stores the RNA var as ATAC_data_var too; nothing reads either.
        self.RNA_data_var, self.ATAC_data_var = RNA_data.var, RNA_data.var
        self.RNA_data, self.ATAC_data = _float32_csr(RNA_data.X), _float32_csr(ATAC_data.X)

    return init


def _float32_csr(matrix):
    """``matrix`` as float32 CSR, without a copy when it already is one."""
    matrix = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
    return matrix.astype(np.float32, copy=False)


class _SparseBatchLoader:
    """Stand-in for the ``DataLoader`` scButterfly builds over its datasets.

    ``DataLoader`` workers assembled every batch from dense per-cell rows on the CPU,
    and the training loop copied it to the GPU. With every target feature modelled, a
    NeurIPS 2022 Multiome cell is 198k floats (23k genes, 175k peaks), so a batch of 64
    is 51 MB and data loading dominated each step: the GPU sat at ~28% and one ATAC
    pretraining epoch took 128 s on a V100. This loader gathers a batch's rows from the
    CSR matrices, ships only their stored entries to the device and scatters them into
    a dense tensor there.

    Batches hold the same values in the same layout (RNA columns, then ATAC), and come
    back on the device as float32, so the training loop's ``.cuda().to(torch.float32)``
    is a no-op. Shuffling draws a ``torch.randperm``, like ``DataLoader``, so it follows
    the seed scButterfly sets.
    """

    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, drop_last=False):
        if hasattr(dataset, "id_list"):  # Single_omics_dataset
            self.blocks = [(dataset.dataset, np.asarray(dataset.id_list))]
        else:  # RNA_ATAC_dataset
            self.blocks = [
                (dataset.RNA_dataset, np.asarray(dataset.id_list_r)),
                (dataset.ATAC_dataset, np.asarray(dataset.id_list_a)),
            ]
        self.n_cells = len(self.blocks[0][1])
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        if self.drop_last:
            return self.n_cells // self.batch_size
        return -(-self.n_cells // self.batch_size)

    def __iter__(self):
        order = torch.randperm(self.n_cells).numpy() if self.shuffle else np.arange(self.n_cells)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            positions = order[start:start + self.batch_size]
            yield torch.cat([self._dense_rows(matrix, cells[positions]) for matrix, cells in self.blocks], dim=1)

    def _dense_rows(self, matrix, cells):
        rows = matrix[cells]
        row_of_entry = np.repeat(np.arange(len(cells)), np.diff(rows.indptr))
        dense = torch.zeros((len(cells), matrix.shape[1]), dtype=torch.float32, device=self.device)
        dense[torch.from_numpy(row_of_entry).to(self.device), torch.from_numpy(rows.indices.astype(np.int64)).to(self.device)] = (
            torch.from_numpy(rows.data).to(self.device)
        )
        return dense


@contextlib.contextmanager
def suppress_unused_postprocessing():
    """Make ``sc.pp.pca``/``sc.pp.neighbors`` no-ops for the duration of the block.

    ``Model.test`` runs both, on *both* predicted matrices, whenever it is not asked
    to draw figures — and there is no flag to turn it off. Nothing reads the results
    back: :func:`predict_cells` only touches ``.X``. On the ATAC side it is a PCA over
    a near-dense ``n_test x n_peaks`` matrix, which dominates the predict step.
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


def read_modality(path, layers=("counts",)):
    """Read an h5ad, keeping only ``layers`` besides ``obs``, ``var`` and ``uns``.

    scButterfly reads only the counts (predict also the target's ``normalized`` layer,
    for the scale map). The NeurIPS 2022 Multiome ATAC files hold two 9 GB layers.
    """
    adata = ad.read_h5ad(path)
    for key in [key for key in adata.layers.keys() if key not in layers]:
        del adata.layers[key]
    for key in list(adata.obsm.keys()):
        del adata.obsm[key]
    return adata


def _to_counts_X(adata):
    """AnnData whose .X is the raw counts layer as float32, carrying obs and var.

    Takes the counts layer out of ``adata``: nothing reads the float64 original again,
    and for the NeurIPS 2022 ATAC input it is 9 GB.

    Counts are stored as float64; scButterfly's model is float32 and its data loader
    does not cast, so feed float32 to avoid a dtype mismatch.

    Everything else is dropped rather than copied. ``sc.concat`` below discards the
    other layers, ``obsm['gene_activity']`` and ``uns`` anyway — the placeholder
    block has none of them and anndata intersects — so a full ``.copy()`` only
    duplicates the largest matrices in the file to throw them away a moment later.
    """
    return ad.AnnData(
        X=adata.layers.pop("counts").astype(np.float32),
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


def validation_cells(n_train):
    """The ~10% of the training cells held out for early stopping (row indices).

    Deterministic, so train and predict hold out the same cells; ``scbutterfly_train``
    also fits the spread of the per-cell scale on them.
    """
    shuffled = list(range(n_train))
    np.random.RandomState(0).shuffle(shuffled)
    return sorted(shuffled[:max(1, int(0.1 * n_train))])


def build_butterfly(train_mod1, train_mod2, test_mod1, n_top_genes, Butterfly,
                    model_all_target_features=True):
    """Build + preprocess + construct a Butterfly for the given data.

    scButterfly's preprocessing keeps the ``n_top_genes`` most variable genes and drops
    peaks open in fewer than 0.5% of the cells. With ``model_all_target_features`` that
    only applies to the input modality: the benchmark scores every target feature, and
    one the model drops could only be predicted as a constant zero. ``False`` reproduces
    models trained before this option existed.

    Consumes the ``counts`` layers of the three inputs (see :func:`_to_counts_X`).

    Returns a dict with the constructed ``butterfly``, the resolved ``direction``,
    ``test_id`` and ``validation_id`` (row indices of the test and held-out training
    cells), ``chrom_list``, and the preprocessed and the original target var names.
    Deterministic given the same inputs, so train and predict produce identical
    architecture/preprocessing.
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
    del rna_train_c, rna_test_c, atac_train_c, atac_test_c

    train_id = list(range(n_train))
    test_id = list(range(n_train, n_train + n_test))

    validation_id = validation_cells(n_train)
    train_id_final = sorted(set(train_id) - set(validation_id))

    # Chromosome ordering (peaks contiguous per chrom + chrom_list).
    sort_index, chrom_list = chrom_utils.sorted_chrom_order(ATAC_data)
    ATAC_data = chrom_utils.apply_sort(ATAC_data, sort_index)

    butterfly = Butterfly()
    butterfly.load_data(RNA_data, ATAC_data, train_id_final, test_id, validation_id)
    del RNA_data, ATAC_data  # load_data keeps its own copies
    butterfly.data_preprocessing(
        n_top_genes=n_top_genes,
        use_hvg=not (model_all_target_features and direction == "ATAC2GEX"),
        filter_features=not (model_all_target_features and direction == "GEX2ATAC"),
    )
    butterfly.augmentation(aug_type=None)
    # Only the preprocessed copies are read from here on.
    butterfly.RNA_data = butterfly.ATAC_data = None

    # scButterfly casts to float32 only on the CUDA path; force float32 for CPU.
    for _attr in ("RNA_data_p", "ATAC_data_p"):
        _adata = getattr(butterfly, _attr, None)
        if _adata is not None and _adata.X is not None:
            _adata.X = _adata.X.astype(np.float32, copy=False)

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
        "validation_id": validation_id,
        "chrom_list": chrom_list,
        "target_p_var_names": list(target_p.var_names),
        "target_var_names": list(train_mod2.var_names),
    }


def predict_cells(built, cell_ids, batch_size, model_path=None):
    """scButterfly's prediction of the target modality for rows ``cell_ids``.

    Returned as a dense float32 array in the target's original var order, still on
    scButterfly's scale (``cell_scale.CellScale`` maps it onto the target's). ``model_path``
    loads the trained weights first; leave it out once they are loaded.
    """
    # Model.test predicts both directions; give the unused one a single cell, which
    # for ATAC -> GEX on NeurIPS 2022 saves a cells x 175k-peak matrix. It also runs
    # PCA + a neighbour graph on both predicted matrices with no way to opt out; none
    # of it is read back. See suppress_unused_postprocessing.
    gex_to_atac = built["direction"] == "GEX2ATAC"
    with suppress_unused_postprocessing():
        A2R_predict, R2A_predict = built["butterfly"].model.test(
            test_id_r=cell_ids if gex_to_atac else cell_ids[:1],
            test_id_a=cell_ids[:1] if gex_to_atac else cell_ids,
            batch_size=batch_size,
            model_path=model_path,
            load_model=model_path is not None,
            test_cluster=False,
            test_figure=False,
            output_data=False,
            return_predict=True,
        )
    prediction = R2A_predict if gex_to_atac else A2R_predict
    # tensor2adata gives integer var names; name them, then scatter by name, which
    # undoes the chromosome sort and zero-fills target features the model dropped.
    prediction.var_names = built["target_p_var_names"]
    n_modelled = len(set(prediction.var_names) & set(built["target_var_names"]))
    logger.info("Modelled target features: %d / %d", n_modelled, len(built["target_var_names"]))
    return chrom_utils.scatter_to_target(prediction, built["target_var_names"])
