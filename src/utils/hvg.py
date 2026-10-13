def hvg_applies(modality):
    """Whether the dataset's `hvg` flag is a usable feature subset for an input of this modality.

    Not for ATAC: the NeurIPS 2021 and 2022 Multiome datasets flag 1,000 of their 116k and 229k peaks as highly
    variable, peaks no more often detected than the rest (median 0.8-0.9% of the cells), and 5,306 of 27,745
    (bmmc_multiome) and 9,068 of 131,095 (pbmc_multiome) cells have no counts in the ones detected in at least
    1% of the cells.
    """
    return modality != "ATAC"
