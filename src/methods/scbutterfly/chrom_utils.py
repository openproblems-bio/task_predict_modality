"""Chromosome / peak utilities for the scButterfly Multiome method.

scButterfly's ``construct_model`` needs a ``chrom_list`` (number of peaks per
chromosome) and reads ``ATAC_data.var.chrom`` during model construction. It also
assumes peaks are contiguous per chromosome. The predict-modality ATAC h5ads have
no ``chrom`` column and peaks are in arbitrary chromosome order, but the peak names
encode the chromosome (``chr17-6651156-6652045`` or ``chr17:6651156-6652045``).

This module parses the chromosome from peak names, produces a peak ordering that
groups peaks contiguously per chromosome (with the matching ``chrom_list``), and
scatters a predicted matrix back into a target var order by feature name.
"""

import itertools
import re

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def parse_chrom(name):
    """Return the contig of a peak name.

    NeurIPS 2021 names peaks ``chr17-6651156-6652045`` and NeurIPS 2022
    ``chr17:6651156-6652045``; both also have peaks on unplaced scaffolds
    (``GL000195.1:32231-33125``), which keep their scaffold as their own group.
    """
    return re.split(r"[:\-]", str(name), maxsplit=1)[0]


def sorted_chrom_order(atac_adata):
    """Compute a peak ordering that groups peaks contiguously by chromosome.

    Returns
    -------
    sort_index : np.ndarray
        Indices that reorder ``atac_adata`` so peaks are grouped per chromosome, in
        sorted chromosome order; the stable sort keeps the original order within one.
    chrom_list : list[int]
        Number of peaks per chromosome, in the sorted order. ``sum == n_peaks``.
    """
    chroms = np.array([parse_chrom(v) for v in atac_adata.var_names])
    sort_index = np.argsort(chroms, kind="stable")
    # np.unique returns the chromosomes sorted, i.e. in the order of sort_index.
    _, chrom_list = np.unique(chroms, return_counts=True)
    return sort_index, chrom_list.tolist()


def chrom_counts(atac_adata):
    """Count peaks per chromosome, in the order they appear in ``var``.

    Assumes peaks are already grouped contiguously per chromosome (as produced by
    :func:`apply_sort`). Reads ``var['chrom']`` if present, else parses from names.
    Peak filtering that preserves order (e.g. scButterfly's TF-IDF/peak filter)
    keeps the grouping contiguous, so recounting here yields a valid ``chrom_list``.
    """
    if "chrom" in atac_adata.var.columns:
        chroms = atac_adata.var["chrom"]
    else:
        chroms = map(parse_chrom, atac_adata.var_names)
    return [len(list(peaks)) for _, peaks in itertools.groupby(chroms)]


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
    target modality's vars (scButterfly sorts peaks by chromosome, and models trained
    before every target feature was modelled subset genes and peaks). Scattering by
    name restores the original order and fills missing target features with zeros.

    Returns a dense float32 ndarray of shape ``(n_cells, len(target_var_names))``.
    """
    X = pred_adata.X.toarray() if issparse(pred_adata.X) else np.asarray(pred_adata.X)
    source_columns = pd.Index(pred_adata.var_names).get_indexer(target_var_names)
    modelled = source_columns >= 0
    out = np.zeros((X.shape[0], len(target_var_names)), dtype=np.float32)
    out[:, modelled] = X[:, source_columns[modelled]]
    return out
