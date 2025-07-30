from typing import Literal
import anndata as ad
import scvi 
from scipy.sparse import issparse


def get_representation(adata: ad.AnnData, modality: Literal["GEX", "ADT", "ATAC"], use_hvg: bool = True) -> ad.AnnData:
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
        - "ATAC": PeakVI model for ATAC data with Bernoulli likelihood on binarized count data. 

        We assume that regardless of the modality, the raw data will be stored in the `counts` layer 
        (e.g. UMI counts for GEX and peak counts for ATAC), and the normalized data in the `normalized` layer.
    use_hvg
        Whether to subset the data to highly variable genes (HVGs) before training the model

    Returns
    -------
    ad.AnnData
        AnnData object with the latent representation in `obsm["X_scvi"]`, regardless of the modality.
    """
    # Subset to highly variable features
    if "hvg" in adata.var.columns and use_hvg:
        n_hvg = adata.var["hvg"].sum()
        print(f"Subsetting to {n_hvg} highly variable features ({n_hvg/adata.n_vars:.2%})", flush=True)
        adata = adata[:, adata.var["hvg"]].copy()
    else:
        print("Training on all available features", flush=True)

    # Setup the AnnData object for scVI
    if modality == "GEX":
        layer = "counts"
        scvi.model.SCVI.setup_anndata(adata, batch_key="batch", layer=layer)
        model = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=30)
    elif modality == "ADT":
        layer = "normalized"
        scvi.model.SCVI.setup_anndata(adata, batch_key="batch", layer=layer)
        model = scvi.model.SCVI(adata, gene_likelihood="normal", n_layers=1, n_latent=10)
    elif modality == "ATAC":
        layer = "counts"
        scvi.model.PEAKVI.setup_anndata(adata, batch_key="batch", layer=layer)
        model = scvi.model.PEAKVI(adata)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    example_data = adata.layers[layer].data if issparse(adata.layers[layer]) else adata.layers[layer]
    print(f"Set up AnnData for modality: {modality}, layer={layer}", flush=True)
    print(f"Data looks like this: {example_data}", flush=True)

    # Train the model
    model.train(early_stopping=True)

    # Get the latent representation
    adata.obsm["X_scvi"] = model.get_latent_representation()

    return adata