"""Helpers shared by ss_opm_train and ss_opm_predict.

ss_opm (https://github.com/shu65/open-problems-multimodal) was written against the Kaggle competition tables and a
set of derived files produced by its `script/make_additional_files.py` and `script/make_cite_input_mask.py`:
z-scored per-cell statistics, z-scored per-batch statistics, a numeric `day`, a donor id (turned into a sex
embedding) and, for CITE-seq, a mask of a few dozen genes whose raw expression is appended to the SVD components.
This module rebuilds all of that from the task's h5ad files, which only guarantee `obs["batch"]` and a
`normalized` layer.
"""

import json
import os
import re
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from sklearn.decomposition import TruncatedSVD

from ss_opm.model.torch_dataset.citeseq_dataset import METADATA_KEYS as CITE_METADATA_KEYS
from ss_opm.model.torch_dataset.multiome_dataset import METADATA_KEYS as MULTI_METADATA_KEYS
from ss_opm.utility.metadata_utility import CELL_TYPES as KAGGLE_CELL_TYPES

# cells per block when densifying a sparse matrix for per-cell or per-batch statistics
ROW_BLOCK = 2000

# number of batch singular-vector columns the CITE model expects (see CITE_METADATA_KEYS)
N_BATCH_SV = sum(key.startswith("batch_sv") for key in CITE_METADATA_KEYS)

CELL_STATISTIC_KEYS = ["nonzero_ratio", "nonzero_q25", "nonzero_q50", "nonzero_q75", "mean", "std"]

# Batch labels of the OpenProblems NeurIPS 2022 datasets are `{day}_{donor}`; the NeurIPS 2021 ones are
# `s{site}d{donor}` and carry no day. Anything that does not match gets a constant.
DEFAULT_DAY_PATTERN = r"^(\d+)_\d+$"
DEFAULT_DONOR_PATTERN = r"^\d+_(\d+)$"

# Reference files the CITE gene masks are built from (same sources as the original `make_cite_input_mask.py`)
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/hgnc_complete_set_2023-01-01.txt"
REACTOME_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"

ENSEMBL_ID_PATTERN = re.compile(r"^ENSG\d+")


def to_sparse_csr(X):
    if scipy.sparse.issparse(X):
        return X.tocsr()
    return scipy.sparse.csr_matrix(X)


def to_dense(X, dtype=np.float32):
    dense = X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)
    return dense.astype(dtype, copy=False)


# ---------------------------------------------------------------------------------------------------------------
# Reference files
# ---------------------------------------------------------------------------------------------------------------
def download_reference_files(directory):
    """Download the HGNC complete set and the Reactome gene sets into `directory` (used at image build time)."""
    os.makedirs(directory, exist_ok=True)
    hgnc_path = os.path.join(directory, "hgnc_complete_set.txt")
    reactome_path = os.path.join(directory, "ReactomePathways.gmt")
    if not os.path.exists(hgnc_path):
        urlretrieve(HGNC_URL, hgnc_path)
    if not os.path.exists(reactome_path):
        archive_path = os.path.join(directory, "ReactomePathways.gmt.zip")
        urlretrieve(REACTOME_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(directory)
        os.remove(archive_path)
    return hgnc_path, reactome_path


def read_reactome_gmt(file_path):
    """Pathway (Reactome stable id) -> gene symbols, from a GMT file (name, id, genes...)."""
    pathway_genes = {}
    with open(file_path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            pathway_genes[fields[1]] = [gene.upper() for gene in fields[2:] if gene]
    return pathway_genes


def read_hgnc(file_path):
    columns = ["symbol", "alias_symbol", "prev_symbol", "ensembl_gene_id"]
    return pd.read_table(file_path, usecols=columns, dtype=str, low_memory=False)


def gene_symbols_from_var_names(var_names, hgnc):
    """Upper-case gene symbols for feature names that are Ensembl ids (`ENSG...` or Kaggle's `ENSG..._SYMBOL`);
    other names (already symbols, e.g. protein names) are kept."""
    ensembl_to_symbol = hgnc.dropna(subset=["ensembl_gene_id"]).set_index("ensembl_gene_id")["symbol"].to_dict()
    symbols = []
    for name in var_names:
        name = str(name)
        if ENSEMBL_ID_PATTERN.match(name):
            if "_" in name:  # Kaggle style: ENSG00000121410_A1BG
                symbols.append(name.split("_", 1)[1].upper())
            else:
                symbols.append(str(ensembl_to_symbol.get(name.split(".")[0], name)).upper())
        else:
            symbols.append(name.upper())
    return np.array(symbols)


def make_targets_gene2idx(target_symbols, hgnc):
    """Port of the original `make_targets_gene2idx`: map every protein name, and every HGNC symbol for which that
    protein name is an alias or previous symbol, to the protein's column index."""
    alias_symbols = {}
    for symbol, aliases, previous in hgnc[["symbol", "alias_symbol", "prev_symbol"]].itertuples(index=False):
        for value in (aliases, previous):
            if pd.isnull(value):
                continue
            for alias in value.upper().split("|"):
                alias_symbols.setdefault(alias, []).append(symbol.upper())

    alias_symbols["CD3"] = ["CD3D", "CD3E", "CD3G"]
    alias_symbols["HLA-A-B-C"] = ["HLA-A", "HLA-B", "HLA-C"]
    alias_symbols["CD45RA"] = ["PTPRC"]
    alias_symbols["CD45RO"] = ["PTPRC"]
    alias_symbols["PODOPLANIN"] = ["PDPN"]
    alias_symbols["HLA-DR"] = [f"HLA-DRB{i}" for i in range(1, 10)] + ["HLA-DRA"]
    alias_symbols["INTEGRINB7"] = ["ITGB7"]
    alias_symbols["CD158"] = ["CD158A"]
    alias_symbols["CD158B"] = ["CD158B1", "CD158B2"]

    targets_gene2idx = {}
    for target_index, target_symbol in enumerate(target_symbols):
        target_symbol = target_symbol.upper()
        targets_gene2idx[target_symbol] = target_index
        for symbol in alias_symbols.get(target_symbol, []):
            targets_gene2idx[symbol] = target_index
    return targets_gene2idx


def _group_spearman(inputs_column, targets_columns):
    """Spearman correlation and p-value of one gene with several targets, over the cells where the gene is
    expressed (as in the original scripts). Returns arrays of NaN when there are too few such cells."""
    expressed = inputs_column > 0
    n_targets = targets_columns.shape[1]
    if expressed.sum() < 3:
        return np.full(n_targets, np.nan), np.full(n_targets, np.nan)
    result = scipy.stats.spearmanr(inputs_column[expressed], targets_columns[expressed])
    correlations = np.atleast_2d(result.statistic)[0, 1:]
    p_values = np.atleast_2d(result.pvalue)[0, 1:]
    return correlations, p_values


def _select_robust_pairs(correlations, min_abs_corr, max_p_value, n_groups):
    """`correlations`: dict (gene index, target index) -> list of (|corr|, p) per group. Keep the pairs that pass
    the thresholds in more than 60 % of the groups and return their median |corr| over the groups that passed,
    mirroring the original code (which stored 0 for failing groups and took the median over all groups)."""
    robust = {}
    for pair, values in correlations.items():
        scores = np.zeros(n_groups)
        for group_index, (corr, p_value) in enumerate(values):
            if np.isfinite(corr) and abs(corr) > min_abs_corr and p_value < max_p_value:
                scores[group_index] = abs(corr)
        if (scores > 0).sum() > 0.6 * n_groups:
            robust[pair] = np.median(scores)
    return robust


def make_cite_input_masks(inputs_lognorm, targets_rownorm, gene_symbols, targets_gene2idx, groups, pathway_genes):
    """Rebuild `cite_inputs_targets_pair3g.npz` and `cite_inputs_mask2.npz` of the original solution.

    Parameters
    ----------
    inputs_lognorm : np.ndarray or scipy.sparse.csr_matrix
        Training RNA, log1p of the median-normalized expression (the original's input transform).
    targets_rownorm : np.ndarray
        Training proteins, row-normalized.
    gene_symbols : np.ndarray
        Upper-case symbol of every input gene.
    targets_gene2idx : dict
        Gene symbol -> protein column index (see `make_targets_gene2idx`).
    groups : np.ndarray
        Group (batch) label of every training cell.
    pathway_genes : dict
        Reactome pathway id -> gene symbols.

    Returns
    -------
    pair_mask : np.ndarray of bool, shape (n_genes, n_proteins)
        For every protein, the gene encoding it (or an alias) whose expression correlates most robustly with it.
    pathway_mask : np.ndarray of bool, shape (n_genes,)
        Up to three genes per protein sharing a Reactome pathway with the protein's gene and correlating with it.
    """
    inputs_lognorm = to_sparse_csr(inputs_lognorm).tocsc()
    n_genes, n_proteins = inputs_lognorm.shape[1], targets_rownorm.shape[1]
    unique_groups = np.unique(groups)
    group_masks = [groups == group for group in unique_groups]

    # candidate (gene, targets) pairs: genes named after a protein, and genes sharing a pathway with such a gene
    direct_candidates = {}
    for gene_index, symbol in enumerate(gene_symbols):
        if symbol in targets_gene2idx:
            direct_candidates[gene_index] = [targets_gene2idx[symbol]]

    pathway_targets = {}
    for genes in pathway_genes.values():
        targets_in_pathway = {targets_gene2idx[gene] for gene in genes if gene in targets_gene2idx}
        if not targets_in_pathway:
            continue
        for gene in genes:
            pathway_targets.setdefault(gene, set()).update(targets_in_pathway)
    pathway_candidates = {
        gene_index: sorted(pathway_targets[symbol])
        for gene_index, symbol in enumerate(gene_symbols)
        if symbol in pathway_targets
    }

    def _correlate(candidates):
        correlations = {}
        for gene_index, target_indices in candidates.items():
            gene_values = inputs_lognorm[:, gene_index].toarray().ravel()
            for group_index, group_mask in enumerate(group_masks):
                corrs, p_values = _group_spearman(gene_values[group_mask], targets_rownorm[group_mask][:, target_indices])
                for target_index, corr, p_value in zip(target_indices, corrs, p_values):
                    correlations.setdefault((gene_index, target_index), []).append((corr, p_value))
        return correlations

    direct_pairs = _select_robust_pairs(_correlate(direct_candidates), 0.10, 1e-2, len(unique_groups))
    pair_mask = np.zeros((n_genes, n_proteins), dtype=bool)
    for target_index in range(n_proteins):
        scored = [(score, gene_index) for (gene_index, t), score in direct_pairs.items() if t == target_index]
        if scored:
            pair_mask[max(scored)[1], target_index] = True

    pathway_pairs = _select_robust_pairs(_correlate(pathway_candidates), 0.20, 1e-3, len(unique_groups))
    pathway_mask = np.zeros(n_genes, dtype=bool)
    for target_index in range(n_proteins):
        scored = sorted(((score, gene_index) for (gene_index, t), score in pathway_pairs.items() if t == target_index), reverse=True)
        for _, gene_index in scored[:3]:
            pathway_mask[gene_index] = True

    return pair_mask, pathway_mask


# ---------------------------------------------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------------------------------------------
def extract_integer_field(batch, pattern):
    """First capture group of `pattern` in every batch label as float, NaN where it does not match."""
    if pattern is None:
        return np.full(len(batch), np.nan)
    return pd.Series(batch).astype(str).str.extract(pattern, expand=False).astype(float).values


def compute_cell_statistics(X, task_type):
    """Per-cell statistics of the normalized expression as in the original `make_*_cell_statistics`: fraction of
    expressed features (log1p of it for multiome), quartiles of the non-zero values, mean and std of the full row."""
    X = to_sparse_csr(X)
    n_features = X.shape[1]
    nonzero_counts = np.diff(X.indptr)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    row_square_sums = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    means = row_sums / n_features
    stds = np.sqrt(np.maximum(row_square_sums / n_features - means**2, 0))

    quartiles = np.zeros((X.shape[0], 3))
    for row, values in enumerate(np.split(X.data, X.indptr[1:-1])):
        if len(values):
            quartiles[row] = np.quantile(values, [0.25, 0.5, 0.75])

    nonzero_ratio = nonzero_counts / n_features
    if task_type == "multi":
        nonzero_ratio = np.log1p(nonzero_ratio)
    return pd.DataFrame(
        {
            "nonzero_ratio": nonzero_ratio,
            "nonzero_q25": quartiles[:, 0],
            "nonzero_q50": quartiles[:, 1],
            "nonzero_q75": quartiles[:, 2],
            "mean": means,
            "std": stds,
        }
    )


def fit_standardization(frame):
    """Mean and std of every column; constant columns get std 1 so that they standardize to 0."""
    means = frame.mean(axis=0)
    stds = frame.std(axis=0, ddof=1).replace(0, 1.0).fillna(1.0)
    return {"mean": means.to_dict(), "std": stds.to_dict()}


def apply_standardization(frame, standardization):
    standardized = frame.copy()
    for column in frame.columns:
        standardized[column] = (frame[column] - standardization["mean"][column]) / standardization["std"][column]
    return standardized


def median_normalized_log_expression(X_block):
    """The original CITE input transform: log1p of every cell divided by its median non-zero expression."""
    dense = np.expm1(to_dense(X_block))
    with np.errstate(invalid="ignore"):
        for_median = np.where(dense == 0, np.nan, dense)
        medians = np.nanmedian(for_median, axis=1)
    medians = np.where(np.isfinite(medians) & (medians > 0), medians, 1.0)
    return np.log1p(dense / medians[:, None])


def compute_batch_input_medians(X, batches):
    """Per batch, the median over its cells of the (median-normalized, log1p) expression of every gene, ignoring
    zeros, as in the original `make_cite_batch_inputs_median`. Returns a DataFrame indexed by batch."""
    X = to_sparse_csr(X)
    batches = np.asarray(batches).astype(str)
    rows = {}
    for batch in np.unique(batches):
        cell_indices = np.flatnonzero(batches == batch)
        values = np.vstack(
            [median_normalized_log_expression(X[cell_indices[start : start + ROW_BLOCK]]) for start in range(0, len(cell_indices), ROW_BLOCK)]
        )
        with np.errstate(invalid="ignore"):
            values[values == 0] = np.nan
            gene_medians = np.nanmedian(values, axis=0)
        gene_medians[~np.isfinite(gene_medians)] = 0.0
        rows[batch] = gene_medians
    return pd.DataFrame.from_dict(rows, orient="index")


class BatchSingularVectors:
    """`batch_sv0..7` of the original CITE metadata: TruncatedSVD of the per-batch gene medians, standardized
    across the batches the model is trained on. Batches first seen at prediction time are projected with the
    fitted components (unknown batches get 0)."""

    def __init__(self, n_components=N_BATCH_SV, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.decomposer = None
        self.standardization = None
        self.table = None

    def fit(self, batch_medians):
        n_components = min(self.n_components, min(batch_medians.shape) - 1) if min(batch_medians.shape) > 1 else 0
        if n_components < 1:
            self.decomposer = None
            self.table = pd.DataFrame(0.0, index=batch_medians.index, columns=self.columns)
            return self
        self.decomposer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        transformed = self.decomposer.fit_transform(batch_medians.values)
        frame = self._to_frame(transformed, batch_medians.index)
        self.standardization = fit_standardization(frame)
        self.table = apply_standardization(frame, self.standardization)
        return self

    def transform(self, batch_medians):
        if self.decomposer is None:
            return pd.DataFrame(0.0, index=batch_medians.index, columns=self.columns)
        frame = self._to_frame(self.decomposer.transform(batch_medians.values), batch_medians.index)
        return apply_standardization(frame, self.standardization)

    @property
    def columns(self):
        return [f"batch_sv{i}" for i in range(self.n_components)]

    def _to_frame(self, transformed, index):
        frame = pd.DataFrame(0.0, index=index, columns=self.columns)
        frame.iloc[:, : transformed.shape[1]] = transformed
        return frame

    def lookup(self, batches):
        """Rows of the batch table for every cell; unknown batches get 0."""
        return self.table.reindex(np.asarray(batches).astype(str)).fillna(0.0).reset_index(drop=True)


def build_metadata(
    batch,
    cell_statistics,
    task_type,
    day_pattern=DEFAULT_DAY_PATTERN,
    donor_pattern=DEFAULT_DONOR_PATTERN,
    batch_sv=None,
    group_by_batch=True,
):
    """The metadata frame ss_opm's datasets and pre-processing expect.

    Parameters
    ----------
    batch : array-like
        Batch label of every cell.
    cell_statistics : pd.DataFrame
        Standardized per-cell statistics (see `compute_cell_statistics`, `apply_standardization`).
    task_type : str
        "cite" or "multi"; the CITE model expects the batch-level columns as well.
    day_pattern, donor_pattern : str or None
        Regex whose first capture group is the day / donor id in a batch label. Cells whose label does not match
        get day 0 and donor -1. The donor only feeds the original's sex embedding, which knows the four Kaggle
        donors; any other donor is embedded like an unknown one.
    batch_sv : pd.DataFrame or None
        Standardized `batch_sv*` columns per cell (see `BatchSingularVectors.lookup`); zeros when None.
    group_by_batch : bool
        Group id used for the per-batch target medians. The predict path has no targets, so it can use one group.
    """
    batch = pd.Series(np.asarray(batch).astype(str))
    metadata = pd.DataFrame(index=range(len(batch)))
    metadata["batch"] = batch.values
    metadata["day"] = np.nan_to_num(extract_integer_field(batch, day_pattern), nan=0.0)
    donor = extract_integer_field(batch, donor_pattern)
    metadata["donor"] = np.where(np.isfinite(donor), donor, -1).astype(int)
    metadata["technology"] = "citeseq" if task_type == "cite" else "multiome"
    # cell types are not part of this task's file format; "hidden" is the label the original used for test cells
    metadata["cell_type"] = "hidden"
    if group_by_batch:
        metadata["group"] = pd.factorize(batch)[0]
    else:
        metadata["group"] = 0

    for key in CELL_STATISTIC_KEYS:
        metadata[key] = cell_statistics[key].values

    if task_type == "cite":
        # Cell-type ratios per batch need cell type labels, which the task does not provide, and the batch cell
        # count is meaningless for a subsampled test set: both are the standardized value of the mean, 0.
        for cell_type in KAGGLE_CELL_TYPES:
            if cell_type != "hidden":
                metadata[f"cell_ratio_{cell_type}"] = 0.0
        metadata["cell_count"] = 0.0
        for column in [f"batch_sv{i}" for i in range(N_BATCH_SV)]:
            metadata[column] = 0.0 if batch_sv is None else batch_sv[column].values

    expected_keys = CITE_METADATA_KEYS if task_type == "cite" else MULTI_METADATA_KEYS
    missing = [key for key in expected_keys if key not in metadata.columns]
    assert not missing, f"metadata is missing {missing}"
    return metadata


# ---------------------------------------------------------------------------------------------------------------
# Training targets and prediction scale
# ---------------------------------------------------------------------------------------------------------------
def informative_cells(targets):
    """Cells whose target vector is not constant. The model is trained with a per-cell correlation loss and
    row-normalizes the targets, both undefined for a constant row (e.g. a cell without any protein counts)."""
    targets = to_sparse_csr(targets)
    n_features = targets.shape[1]
    row_sums = np.asarray(targets.sum(axis=1)).ravel()
    row_square_sums = np.asarray(targets.multiply(targets).sum(axis=1)).ravel()
    variances = row_square_sums / n_features - (row_sums / n_features) ** 2
    return variances > 1e-12


def fit_prediction_rescaling(predictions, targets):
    """The model predicts per-cell z-scores. One global affine map to the target scale keeps every per-cell and
    per-feature correlation unchanged and makes the RMSE/MAE metrics meaningful."""
    slope, intercept = np.polyfit(np.asarray(predictions, dtype=np.float64).ravel(), to_dense(targets, np.float64).ravel(), deg=1)
    return {"slope": float(slope), "intercept": float(intercept)}


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------------------------------------------
# Runtime patches of the ss_opm package
# ---------------------------------------------------------------------------------------------------------------
def _safe_median_normalize(values, ignore_zero=True, log=False):
    """`median_normalize` of ss_opm, keeping the input dtype (the original upcasts float32 data to float64, which
    doubles the memory of the dense multiome targets) and leaving rows with an undefined or zero median unchanged."""
    dense = to_dense(values, dtype=values.dtype if isinstance(values, np.ndarray) else np.float32)
    medians = np.empty(dense.shape[0], dtype=np.float64)
    for start in range(0, dense.shape[0], ROW_BLOCK):
        block = dense[start : start + ROW_BLOCK].astype(np.float64)
        if ignore_zero:
            block[block == 0] = np.nan
        with np.errstate(invalid="ignore"):
            medians[start : start + ROW_BLOCK] = np.nanquantile(block, q=0.5, axis=1)
    medians = np.where(np.isfinite(medians) & (medians != 0), medians, 1.0).astype(dense.dtype)
    if log:
        return dense - medians[:, None]
    return dense / medians[:, None]


def _safe_row_normalize(values):
    """`row_normalize` of ss_opm; constant rows become zeros instead of NaN."""
    means = np.mean(values, axis=1, keepdims=True)
    stds = np.std(values, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (values - means) / stds


def _safe_row_quantile_normalize(values, q=0.5):
    """`row_quantile_normalize` of ss_opm without mutating its input and skipping rows without non-zero values."""
    values = values.tocsr()
    normalized_data = values.data.astype(np.float32, copy=True)
    for row, (start, end) in enumerate(zip(values.indptr[:-1], values.indptr[1:])):
        if end > start:
            quantile = np.quantile(normalized_data[start:end], q=q)
            if quantile > 0:
                normalized_data[start:end] /= quantile
    return scipy.sparse.csr_matrix((normalized_data, values.indices, values.indptr), values.shape)


def apply_runtime_patches():
    """Swap in the dtype-preserving, zero-row-safe normalizers for every caller inside ss_opm.

    Train and predict run the same preprocessing chain, so both have to call this before `PrePostProcessing`.
    """
    import ss_opm.pre_post_processing.pre_post_processing as pre_post_processing_module
    import ss_opm.utility.nonzero_median_normalize as median_normalize_module
    import ss_opm.utility.row_normalize as row_normalize_module

    median_normalize_module.median_normalize = _safe_median_normalize
    median_normalize_module.row_quantile_normalize = _safe_row_quantile_normalize
    row_normalize_module.row_normalize = _safe_row_normalize
    # names already bound inside pre_post_processing's namespace by `from X import Y`
    pre_post_processing_module.median_normalize = _safe_median_normalize
    pre_post_processing_module.row_quantile_normalize = _safe_row_quantile_normalize
    pre_post_processing_module.row_normalize = _safe_row_normalize
