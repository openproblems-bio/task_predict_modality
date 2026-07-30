"""Chromosome / peak utilities for the scButterfly Multiome method.

scButterfly's ``construct_model`` needs a ``chrom_list`` (number of peaks per
chromosome) and reads ``ATAC_data.var.chrom`` during model construction. It also
assumes peaks are contiguous per chromosome. The predict-modality ATAC h5ads have
no ``chrom`` column and peaks are in arbitrary chromosome order, but the peak names
encode the chromosome (e.g. ``chr17-6651156-6652045``).

This module parses the chromosome from peak names, produces a peak ordering that
groups peaks contiguously per chromosome (with the matching ``chrom_list``), and
scatters a predicted matrix back into a target var order by feature name.
"""

import re

import numpy as np
from scipy.sparse import issparse

# Matches a leading chromosome token like "chr17", "chrX", "17", "X" followed by
# a ':' or '-' delimiter. Used as a fallback when the simple split does not yield
# a recognisable chromosome.
_CHROM_RE = re.compile(r"^chr?([0-9XYMxym]+)[:\-]")


def parse_chrom(name):
    """Return the chromosome token for a peak name (e.g. 'chr17')."""
    name = str(name)
    token = name.split("-", 1)[0]
    if token.lower().startswith("chr"):
        return token
    m = _CHROM_RE.match(name)
    if m:
        return "chr" + m.group(1)
    # Unparseable — bucket everything unknown together so it still forms one group.
    return token


def sorted_chrom_order(atac_adata):
    """Compute a peak ordering that groups peaks contiguously by chromosome.

    Returns
    -------
    sort_index : np.ndarray
        Indices that reorder ``atac_adata`` so peaks are grouped per chromosome.
    chrom_list : list[int]
        Number of peaks per chromosome, in the sorted order. ``sum == n_peaks``.
    """
    chroms = np.array([parse_chrom(v) for v in atac_adata.var_names])
    # Stable ordering of chromosomes; stable argsort keeps peaks deterministic
    # within a chromosome (preserving original relative order).
    chrom_order = sorted(set(chroms))
    rank = {c: i for i, c in enumerate(chrom_order)}
    keys = np.array([rank[c] for c in chroms])
    sort_index = np.argsort(keys, kind="stable")

    sorted_chroms = chroms[sort_index]
    chrom_list = []
    last = None
    for c in sorted_chroms:
        if c != last:
            chrom_list.append(1)
            last = c
        else:
            chrom_list[-1] += 1
    assert sum(chrom_list) == atac_adata.n_vars
    return sort_index, chrom_list


def chrom_counts(atac_adata):
    """Count peaks per chromosome, in the order they appear in ``var``.

    Assumes peaks are already grouped contiguously per chromosome (as produced by
    :func:`apply_sort`). Reads ``var['chrom']`` if present, else parses from names.
    Peak filtering that preserves order (e.g. scButterfly's TF-IDF/peak filter)
    keeps the grouping contiguous, so recounting here yields a valid ``chrom_list``.
    """
    if "chrom" in atac_adata.var.columns:
        chroms = list(atac_adata.var["chrom"])
    else:
        chroms = [parse_chrom(v) for v in atac_adata.var_names]
    counts = []
    last = None
    for c in chroms:
        if c != last:
            counts.append(1)
            last = c
        else:
            counts[-1] += 1
    return counts


def apply_sort(atac_adata, sort_index):
    """Reorder ATAC peaks by ``sort_index`` and set ``.var['chrom']``.

    Returns a new AnnData whose peaks are contiguous per chromosome and which
    carries the parsed chromosome in ``var['chrom']`` for scButterfly to read.
    """
    out = atac_adata[:, sort_index].copy()
    out.var["chrom"] = [parse_chrom(v) for v in out.var_names]
    return out


def scatter_to_target(pred_adata, target_var_names):
    """Scatter a prediction into ``target_var_names`` order, by feature name.

    Predicted features may be a subset of and/or in a different order from the
    target modality's vars (scButterfly can subset RNA to HVGs or filter ATAC
    peaks). Scattering by name simultaneously (a) restores the original peak
    order and (b) fills any missing target features with zeros.

    Returns a dense float32 ndarray of shape ``(n_cells, len(target_var_names))``.
    """
    X = pred_adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n_cells = X.shape[0]
    out = np.zeros((n_cells, len(target_var_names)), dtype=np.float32)

    target_pos = {name: i for i, name in enumerate(target_var_names)}
    for src_col, name in enumerate(pred_adata.var_names):
        dst = target_pos.get(name)
        if dst is not None:
            out[:, dst] = X[:, src_col]
    return out
