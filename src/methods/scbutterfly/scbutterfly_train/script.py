import logging
import os
import pickle
import sys

from scipy.sparse import csr_matrix

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
from cell_scale import CellScale

butterfly_common.apply_runtime_patches()
from scButterfly.butterfly import Butterfly

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
logger.info("Reading input files...")
train_mod1 = butterfly_common.read_modality(par["input_train_mod1"], layers=("counts", "normalized"))
train_mod2 = butterfly_common.read_modality(par["input_train_mod2"], layers=("counts", "normalized"))
test_mod1 = butterfly_common.read_modality(par["input_test_mod1"])

# ---------------------------------------------------------------------------
# Per-cell scale of the predictions (src/utils/cell_scale.py, as in ss_opm and
# senkin_tmp). scButterfly predicts its own preprocessing of the target, not the
# target's normalized layer; CellScale gives every predicted cell the level and spread
# of the target. Its ridge model is fitted on all training cells here, its spread on
# the cells held out for early stopping once the model is trained. build_butterfly
# takes the counts out of the inputs, so slice what the spread needs first.
# ---------------------------------------------------------------------------
logger.info("Fitting the per-cell scale model...")
train_inputs = csr_matrix(train_mod1.layers.pop("normalized"))
train_targets = csr_matrix(train_mod2.layers.pop("normalized"))
train_counts = csr_matrix(train_mod1.layers["counts"])
cell_scale = CellScale().fit(train_inputs, train_targets, train_counts)
validation_id = butterfly_common.validation_cells(train_mod1.n_obs)
validation_inputs, validation_targets, validation_counts = (
    matrix[validation_id] for matrix in (train_inputs, train_targets, train_counts)
)
del train_inputs, train_targets, train_counts

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

# Spread of the per-cell scale, from the held-out cells predicted with the weights
# train_model saved, which are the ones scbutterfly_predict loads.
validation_predictions = butterfly_common.predict_cells(
    built, validation_id, par["batch_size"], model_path=par["output"],
)
cell_scale.fit_spread(validation_predictions, validation_inputs, validation_targets, validation_counts)
logger.info("Per-cell scale: spread shrinkage %.4f", cell_scale.spread_)
cell_scale.save(os.path.join(par["output"], "cell_scale.npz"))

# ---------------------------------------------------------------------------
# Persist the metadata predict needs to reconstruct the model deterministically.
# The trained weights live in <output>/model/*.pt (written by train_model), the
# per-cell scale in <output>/cell_scale.npz.
# ---------------------------------------------------------------------------
logger.info("Saving model metadata...")
metadata = {
    "direction": built["direction"],
    "n_top_genes": par["n_top_genes"],
    "model_all_target_features": True,
    "batch_size": par["batch_size"],
    "target_var_names": list(train_mod2.var_names),
    "dataset_id": train_mod1.uns.get("dataset_id", ""),
}
with open(os.path.join(par["output"], "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f, protocol=4)

logger.info("Training complete. Model saved to %s", par["output"])
