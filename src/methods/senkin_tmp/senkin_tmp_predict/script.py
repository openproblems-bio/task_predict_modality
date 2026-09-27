import logging
import os
import pickle

import anndata as ad
import pandas as pd
from scipy.sparse import csc_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/test_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod2.h5ad",
    "input_model": "output/models/senkin_tmp",
    "output": "output_pred.h5ad",
}
meta = {"name": "senkin_tmp"}
## VIASH END


def read_model_bundle(model_dir):
    """The train step already predicts the test cells (the solution is transductive) and stores them as the AnnData
    `predictions.h5ad`. Bundles written before that were a pickle of pandas objects, which the pandas of this image
    cannot always unpickle; they are still read so that the existing test resources keep working until they are
    regenerated."""
    h5ad_path = os.path.join(model_dir, "predictions.h5ad")
    if os.path.exists(h5ad_path):
        return ad.read_h5ad(h5ad_path)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as handle:
        legacy = pickle.load(handle)
    test_obs_names = legacy["test_obs_names"] if "test_obs_names" in legacy else legacy["test_obs"].index
    return ad.AnnData(
        layers={"normalized": legacy["test_predictions"]},
        obs=pd.DataFrame(index=pd.Index(test_obs_names).astype(str)),
        var=legacy["prot_var"],
        uns={"dataset_id": legacy.get("dataset_id", "")},
    )


logger.info("Reading input files...")
adata_rna_test = ad.read_h5ad(par["input_test_mod1"])
adata_prot_train = ad.read_h5ad(par["input_train_mod2"])

logger.info("Loading model bundle...")
bundle = read_model_bundle(par["input_model"])
# The train step predicted the test cells of input_test_mod1 in their order; make sure nothing changed
assert bundle.obs_names.equals(adata_rna_test.obs_names), "test cells do not match the trained model"
assert bundle.var_names.equals(adata_prot_train.var_names), "proteins do not match the trained model"

logger.info("Writing predictions...")
adata_out = ad.AnnData(
    layers={"normalized": csc_matrix(bundle.layers["normalized"])},
    obs=adata_rna_test.obs,
    var=adata_prot_train.var,
    uns={
        "dataset_id": adata_rna_test.uns.get("dataset_id", bundle.uns.get("dataset_id", "")),
        "method_id": "senkin_tmp",
    },
)

adata_out.write_h5ad(par["output"], compression="gzip")
logger.info("Predictions saved to %s", par["output"])
