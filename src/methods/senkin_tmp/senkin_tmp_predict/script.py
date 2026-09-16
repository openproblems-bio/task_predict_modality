import logging
import pickle

import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/test_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod2.h5ad",
    "input_model": "output_model.pkl",
    "output": "output_pred.h5ad",
}
meta = {"name": "senkin_tmp"}
## VIASH END

logger.info("Reading input files...")
adata_rna_test   = ad.read_h5ad(par["input_test_mod1"])
adata_prot_train = ad.read_h5ad(par["input_train_mod2"])

logger.info("Loading model bundle...")
with open(par["input_model"], "rb") as f:
    bundle = pickle.load(f)

predictions = bundle["test_predictions"]
if "test_obs_names" in bundle:
    # The train step predicted the test cells in the order of input_test_mod1; make sure nothing changed
    assert (np.asarray(bundle["test_obs_names"]) == adata_rna_test.obs_names.values).all(), "test cells do not match the trained model"
assert predictions.shape == (adata_rna_test.n_obs, adata_prot_train.n_vars)

logger.info("Writing predictions...")
adata_out = ad.AnnData(
    layers={"normalized": csc_matrix(predictions)},
    obs=adata_rna_test.obs,
    var=adata_prot_train.var,
    uns={
        "dataset_id": adata_rna_test.uns.get("dataset_id", bundle.get("dataset_id", "")),
        "method_id": "senkin_tmp",
    },
)

adata_out.write_h5ad(par["output"], compression="gzip")
logger.info("Predictions saved to %s", par["output"])
