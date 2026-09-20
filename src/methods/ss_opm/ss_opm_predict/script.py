import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
import torch

## VIASH START
par = {
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad",
    "input_model": "output/models/ss_opm",
    "output": "output/prediction.h5ad",
}
meta = {"name": "ss_opm_predict", "resources_dir": "src/methods/ss_opm"}
## VIASH END

sys.path.append(meta["resources_dir"])
from ss_opm_common import (  # noqa: E402
    apply_runtime_patches,
    apply_standardization,
    build_metadata,
    compute_cell_statistics,
    load_model_bundle,
    to_sparse_csr,
)

apply_runtime_patches()

from ss_opm.model.encoder_decoder.encoder_decoder import EncoderDecoder  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# ---- Load the training bundle ----
task_info, batch_singular_vectors = load_model_bundle(par["input_model"])
task_type = task_info["task_type"]
print(f"Task type: {task_type}, {task_info['mod1']} -> {task_info['mod2']}", flush=True)

with open(os.path.join(par["input_model"], "pre_post_process.pickle"), "rb") as handle:
    pre_post_process = pickle.load(handle)
mod2_var = pd.read_parquet(os.path.join(par["input_model"], "mod2_var.parquet"))

model = EncoderDecoder(params=None)
# PyTorch >= 2.6 defaults to weights_only=True, which rejects the pickled module classes of this trusted local file
original_torch_load = torch.load
torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, "weights_only": False})
model.load(os.path.join(par["input_model"], "model"))
torch.load = original_torch_load
model.params["device"] = device
model.model.float()

# ---- Test data and its metadata, standardized with the training statistics ----
print("Loading test data...", flush=True)
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])
test_inputs = to_sparse_csr(input_test_mod1.layers["normalized"]).astype(np.float32)
test_batches = input_test_mod1.obs["batch"].astype(str).values
test_cell_statistics = apply_standardization(
    compute_cell_statistics(test_inputs, task_type), task_info["cell_statistics_standardization"]
)
test_metadata = build_metadata(
    test_batches,
    test_cell_statistics,
    task_type=task_type,
    day_pattern=task_info["day_pattern"],
    donor_pattern=task_info["donor_pattern"],
    batch_sv=batch_singular_vectors.lookup(test_batches) if batch_singular_vectors is not None else None,
    group_by_batch=False,
)

# ---- Predict ----
print("Preprocessing test data...", flush=True)
preprocessed_test_inputs, _ = pre_post_process.preprocess(inputs_values=test_inputs, targets_values=None, metadata=test_metadata)
preprocessed_test_inputs = np.asarray(preprocessed_test_inputs, dtype=np.float32)

print("Predicting...", flush=True)
predictions = model.predict(x=test_inputs, preprocessed_x=preprocessed_test_inputs, metadata=test_metadata)
rescaling = task_info["prediction_rescaling"]
predictions = predictions * rescaling["slope"] + rescaling["intercept"]
predictions = np.nan_to_num(predictions, nan=rescaling["intercept"], posinf=rescaling["intercept"], neginf=rescaling["intercept"])
assert predictions.shape == (input_test_mod1.n_obs, mod2_var.shape[0])

# ---- Write ----
print("Writing output...", flush=True)
output = ad.AnnData(
    layers={"normalized": scipy.sparse.csr_matrix(predictions.astype(np.float32))},
    obs=input_test_mod1.obs,
    var=mod2_var,
    uns={"dataset_id": task_info["dataset_id"], "method_id": "ss_opm"},
)
output.write_h5ad(par["output"], compression="gzip")
print("Done!", flush=True)
