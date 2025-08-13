from typing import Literal
import anndata as ad
import scvi 
from scipy.sparse import issparse, csr_matrix, csc_matrix
import muon
import scanpy as sc
import numpy as np



def preprocess_features(
    adata: ad.AnnData,
    modality: Literal["GEX", "ADT", "ATAC"],
    use_hvg: bool = True,
    min_cells_fraction: float = 0.01,
    feature_filter_threshold: int = 20000,
) -> ad.AnnData:
    """
    Preprocess features with optional filtering and HVG selection.
    
    Parameters
    ----------
    adata
        AnnData object to preprocess
    modality
        The modality type, affects filtering strategy
    use_hvg
        Whether to apply HVG selection (applies to all modalities if requested)
    min_cells_fraction
        Minimum fraction of cells a feature must be detected in to be kept
    feature_filter_threshold
        Only apply feature filtering if n_vars > this threshold
        
    Returns
    -------
    Preprocessed AnnData object
    """
    print(f"Starting preprocessing for {modality} with {adata.n_obs} cells and {adata.n_vars} features")
    
    # Feature filtering: only apply if we have many features (mainly for ATAC)
    if adata.n_vars > feature_filter_threshold:
        min_cells = int(adata.n_obs * min_cells_fraction)
        print(f"Applying feature filtering: removing features detected in <{min_cells} cells ({min_cells_fraction:.1%} threshold)")
        
        n_features_before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cells)
        n_features_after = adata.n_vars
        
        print(f"Features after filtering: {n_features_after} (removed {n_features_before - n_features_after})")
    else:
        print(f"Skipping feature filtering (only {adata.n_vars} features, threshold is {feature_filter_threshold})")
    
    # HVG selection: apply if requested
    if use_hvg:
        n_hvg = adata.var["hvg"].sum()
        print(f"Applying HVG selection: subsetting to {n_hvg} highly variable features ({n_hvg/adata.n_vars:.2%})")
        adata = adata[:, adata.var["hvg"]].copy()
    else:
        print("HVG selection disabled, using all remaining features")
    
    # Critical check: ensure no cells have zero counts after filtering
    cell_counts = np.array(adata.layers['counts'].sum(axis=1)).flatten()
    zero_count_cells = (cell_counts == 0).sum()
    
    if zero_count_cells > 0:
        raise ValueError(
            f"After preprocessing, {zero_count_cells} cells have zero counts! "
            "This would prevent generating latent representations for these cells. "
            "Consider relaxing filtering parameters or checking data quality."
        )
    
    print(f"Final dataset: {adata.n_obs} cells × {adata.n_vars} features")
    print(f"Mean counts per cell: {cell_counts.mean():.1f}")
    
    return adata


def get_representation(
        adata: ad.AnnData, 
        modality: Literal["GEX", "ADT", "ATAC"], 
        use_hvg: bool = True, 
        adt_normalization: Literal["clr", "log_cp10k"] = "clr",
        plot_umap: bool = False,
        min_cells_fraction: float = 0.05,
        feature_filter_threshold: int = 20000,
    ) -> ad.AnnData:
    """
    Get a joint latent space representation of the data based on the modality.
    
    Parameters
    ----------
    adata
        AnnData object containing concateneted train and test data from the same modality. 
    modality
        The modality of the data, one of "GEX", "ADT", or "ATAC". Depeding on the modality, we fit the following models:

        - "GEX": scVI model for gene expression data with ZINB likelihood on raw counts. 
        - "ADT": scVI model for ADT data (surface proteins) with Gaussian likelihood on normalized data.
        - "ATAC": PeakVI model for ATAC data with Bernoulli likelihood on count data. 

        We assume that regardless of the modality, the raw data will be stored in the `counts` layer 
        (e.g. UMI counts for GEX and peak counts for ATAC), and the normalized data in the `normalized` layer.
    use_hvg
        Whether to subset the data to highly variable genes (HVGs) before training the model
    adt_normalization
        Normalization method for ADT data. Options are:
         - "clr" (centered log-ratio transformation)
         - "log_cp10k" (normalization to 10k counts per cell and logarithm transformation)
    plot_umap
        Purely for diagnostic purposes, to see whether the data integration looks ok, this optionally computes 
        a UMAP in shared latent space and stores a plot.
    min_cells_fraction
        Minimum fraction of cells a feature must be detected in to be kept during filtering.
    feature_filter_threshold
        Only apply feature filtering if n_vars > this threshold. This automatically separates
        ATAC data (many peaks) from other modalities (fewer features).

    Returns
    -------
    ad.AnnData
        AnnData object with the latent representation in `obsm["X_scvi"]`, regardless of the modality.
    """
    # Preprocess features (filtering and HVG selection)
    adata = preprocess_features(
        adata, 
        modality=modality, 
        use_hvg=use_hvg,
        min_cells_fraction=min_cells_fraction,
        feature_filter_threshold=feature_filter_threshold
    )

    # Setup the AnnData object for scVI
    if modality == "GEX":
        layer = "counts"
        scvi.model.SCVI.setup_anndata(adata, layer=layer, categorical_covariate_keys=["split", "batch"])
        model = scvi.model.SCVI(adata)

    elif modality == "ADT":
        print(f"Normalizing the ADT data using method '{adt_normalization}'")
        if adt_normalization == "clr":
            adata.X = csc_matrix(adata.layers["counts"]) # Use raw counts for ADT
            muon.prot.pp.clr(adata)
            adata.layers["adt_normalized"] = csr_matrix(adata.X)
        elif adt_normalization == "log_cp10k":
            adata.layers["adt_normalized"] = adata.layers["normalized"]
        else:
            raise ValueError(f"Unknown ADT normalization method: {adt_normalization}")
        
        layer = "adt_normalized"
        scvi.model.SCVI.setup_anndata(adata, layer=layer, categorical_covariate_keys=["split", "batch"])
        model = scvi.model.SCVI(adata, gene_likelihood="normal", n_layers=1, n_latent=10)
    elif modality == "ATAC":

        print("Converting read counts to fragment counts")
        scvi.data.reads_to_fragments(adata, read_layer="counts")
        print(f"One counts: {(adata.layers['fragments'] == 1).sum()}, Two counts: {(adata.layers['fragments'] == 2).sum()}")
        layer = "fragments"

        scvi.external.POISSONVI.setup_anndata(adata, layer=layer, categorical_covariate_keys=["split", "batch"])
        model = scvi.external.POISSONVI(adata)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    example_data = adata.layers[layer].data if issparse(adata.layers[layer]) else adata.layers[layer]
    print(f"Set up AnnData for modality: '{modality}' using layer: '{layer}'", flush=True)
    print(f"Data looks like this: {example_data}", flush=True)
    print(model, flush=True)

    # Train the model
    model.train(early_stopping=True)

    # Get the latent representation
    adata.obsm["X_scvi"] = model.get_latent_representation()

    if plot_umap:
        sc.pp.neighbors(adata, use_rep="X_scvi")
        sc.tl.umap(adata)

        plot_name = f"_{modality}_{adt_normalization}_use_hvg_{use_hvg}.png"
        sc.pl.embedding(adata, basis="umap", color=["batch", "split"], show=False, save=plot_name)

    return adata