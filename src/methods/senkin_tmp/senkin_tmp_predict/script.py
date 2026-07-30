import logging
import pickle

import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod2.h5ad",
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

logger.info("Writing predictions...")
adata_out = ad.AnnData(
    layers={"normalized": csc_matrix(bundle["test_predictions"])},
    obs=adata_rna_test.obs,
    var=adata_prot_train.var,
    uns={
        "dataset_id": adata_rna_test.uns.get("dataset_id", bundle.get("dataset_id", "")),
        "method_id": "senkin_tmp",
    },
)

adata_out.write_h5ad(par["output"], compression="gzip")
logger.info("Predictions saved to %s", par["output"])
