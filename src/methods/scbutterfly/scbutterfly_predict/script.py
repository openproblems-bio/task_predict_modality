import logging
import os
import pickle
import sys

import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_train_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/train_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/train_mod2.h5ad",
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/test_mod1.h5ad",
    "input_model": "output_model",
    "output": "output_pred.h5ad",
}
meta = {"name": "scbutterfly", "resources_dir": "src/methods/scbutterfly"}
## VIASH END

sys.path.append(meta["resources_dir"])
import butterfly_common
from cell_scale import CellScale

butterfly_common.apply_runtime_patches()
from scButterfly.butterfly import Butterfly

# ---------------------------------------------------------------------------
# Load data + model metadata
# ---------------------------------------------------------------------------
logger.info("Reading input files...")
train_mod1 = butterfly_common.read_modality(par["input_train_mod1"])
train_mod2 = butterfly_common.read_modality(par["input_train_mod2"])
test_mod1 = butterfly_common.read_modality(par["input_test_mod1"], layers=("counts", "normalized"))
# The per-cell scale reads the test inputs; build_butterfly takes the counts out of test_mod1.
test_inputs, test_counts = test_mod1.layers.pop("normalized"), test_mod1.layers["counts"]

with open(os.path.join(par["input_model"], "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

# ---------------------------------------------------------------------------
# Reconstruct the SAME model architecture + preprocessing as train, then load the
# trained weights (written by train_model to <model>/model/*.pt) and run inference.
# ---------------------------------------------------------------------------
logger.info("Reconstructing scButterfly model...")
built = butterfly_common.build_butterfly(
    train_mod1, train_mod2, test_mod1,
    n_top_genes=metadata["n_top_genes"], Butterfly=Butterfly,
    # Models trained before every target feature was modelled lack the key.
    model_all_target_features=metadata.get("model_all_target_features", False),
)
batch_size = metadata["batch_size"]

logger.info("Loading trained weights and predicting...")
test_predictions = butterfly_common.predict_cells(
    built, built["test_id"], batch_size, model_path=par["input_model"],
)

# scButterfly predicts its own preprocessing of the target; give every cell the level
# and spread of the target's normalized layer (fitted in scbutterfly_train).
cell_scale_path = os.path.join(par["input_model"], "cell_scale.npz")
if os.path.exists(cell_scale_path):
    test_predictions = CellScale.load(cell_scale_path).apply(test_predictions, test_inputs, test_counts)
else:
    logger.warning("The model has no cell_scale.npz (trained before the per-cell scale): writing scButterfly's own output.")

# ---------------------------------------------------------------------------
# Write predictions.
# ---------------------------------------------------------------------------
# viash sets meta["name"] to the component name (scbutterfly_predict); report the
# method name expected by the benchmark by stripping the _predict/_train suffix.
method_id = meta["name"]
for _suffix in ("_predict", "_train"):
    if method_id.endswith(_suffix):
        method_id = method_id[: -len(_suffix)]
        break

logger.info("Writing predictions...")
adata_out = ad.AnnData(
    layers={"normalized": csc_matrix(test_predictions.astype(np.float32))},
    obs=test_mod1.obs,
    var=train_mod2.var,
    uns={
        "dataset_id": test_mod1.uns.get("dataset_id", metadata.get("dataset_id", "")),
        "method_id": method_id,
    },
)

adata_out.write_h5ad(par["output"], compression="gzip")
logger.info("Predictions saved to %s", par["output"])
