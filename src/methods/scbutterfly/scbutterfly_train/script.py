import logging
import os
import pickle
import sys

import anndata as ad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_train_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/train_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/train_mod2.h5ad",
    "input_test_mod1":  "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/test_mod1.h5ad",
    "output": "output_model",
    "rna_pretrain_epoch": 100,
    "atac_pretrain_epoch": 100,
    "translator_epoch": 200,
    "patience": 50,
    "batch_size": 64,
    "n_top_genes": 3000,
}
meta = {"name": "scbutterfly", "resources_dir": "src/methods/scbutterfly"}
## VIASH END

sys.path.append(meta["resources_dir"])
import butterfly_common

butterfly_common.apply_runtime_patches()
from scButterfly.butterfly import Butterfly

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
logger.info("Reading input files...")
train_mod1 = ad.read_h5ad(par["input_train_mod1"])
train_mod2 = ad.read_h5ad(par["input_train_mod2"])
test_mod1 = ad.read_h5ad(par["input_test_mod1"])

# ---------------------------------------------------------------------------
# Build + construct the model, then train (weights written to <output>/model).
# ---------------------------------------------------------------------------
os.makedirs(par["output"], exist_ok=True)

logger.info("Building scButterfly model...")
built = butterfly_common.build_butterfly(
    train_mod1, train_mod2, test_mod1,
    n_top_genes=par["n_top_genes"], Butterfly=Butterfly,
)
butterfly = built["butterfly"]

logger.info("Training scButterfly model...")
butterfly.train_model(
    R2R_pretrain_epoch=par["rna_pretrain_epoch"],
    A2A_pretrain_epoch=par["atac_pretrain_epoch"],
    translator_epoch=par["translator_epoch"],
    patience=par["patience"],
    batch_size=par["batch_size"],
    output_path=par["output"],
)

# ---------------------------------------------------------------------------
# Persist the metadata predict needs to reconstruct the model deterministically.
# The trained weights live in <output>/model/*.pt (written by train_model).
# ---------------------------------------------------------------------------
logger.info("Saving model metadata...")
metadata = {
    "direction": built["direction"],
    "n_top_genes": par["n_top_genes"],
    "batch_size": par["batch_size"],
    "target_var_names": list(train_mod2.var_names),
    "dataset_id": train_mod1.uns.get("dataset_id", ""),
}
with open(os.path.join(par["output"], "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f, protocol=4)

logger.info("Training complete. Model saved to %s", par["output"])
