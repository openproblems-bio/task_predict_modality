import gc
import os
import pickle
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import torch

## VIASH START
par = {
    "input_train_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod2.h5ad",
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad",
    "output": "output/models/ss_opm",
    "day_pattern": r"^(\d+)_\d+$",
    "donor_pattern": r"^\d+_(\d+)$",
    "hgnc_complete_set": None,
    "reactome_pathways": None,
    "n_epochs": 40,
    "burnin_length_epoch": 10,
    "n_rescaling_cells": 5000,
}
meta = {"name": "ss_opm_train", "resources_dir": "src/methods/ss_opm", "cpus": None}
## VIASH END

sys.path.append(meta["resources_dir"])
from ss_opm_common import (  # noqa: E402
    ROW_BLOCK,
    BatchSingularVectors,
    apply_runtime_patches,
    apply_standardization,
    build_metadata,
    compute_batch_input_medians,
    compute_cell_statistics,
    download_reference_files,
    fit_prediction_rescaling,
    fit_standardization,
    gene_symbols_from_var_names,
    informative_cells,
    make_cite_input_masks,
    make_targets_gene2idx,
    median_normalized_log_expression,
    read_hgnc,
    read_reactome_gmt,
    save_json,
    to_dense,
    to_sparse_csr,
)

apply_runtime_patches()

from ss_opm.model.encoder_decoder.encoder_decoder import EncoderDecoder  # noqa: E402
from ss_opm.pre_post_processing.pre_post_processing import PrePostProcessing  # noqa: E402
from ss_opm.utility.row_normalize import row_normalize  # noqa: E402
from ss_opm.utility.set_seed import set_seed  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# The SVD components are stored as float64 tensors inside the torch module while the network is float32:
# cast the module before every forward pass (a no-op after the first one).
import ss_opm.model.encoder_decoder.encoder_decoder as encoder_decoder_module  # noqa: E402

_original_train_step_forward = encoder_decoder_module.EncoderDecoder._train_step_forward


def _float32_train_step_forward(self, batch, training_length_ratio):
    if getattr(self, "model", None) is not None:
        self.model.float()
    return _original_train_step_forward(self, batch, training_length_ratio)


encoder_decoder_module.EncoderDecoder._train_step_forward = _float32_train_step_forward

# The DataLoader spawns one worker per OMP_NUM_THREADS (the original's convention); keep it within the allocation
if meta.get("cpus"):
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(int(meta["cpus"]), 8))))

SEED = 42
set_seed(SEED)

# ---- Load data ----
print("Loading data...", flush=True)
input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"]) if par.get("input_test_mod1") else None

mod1 = input_train_mod1.uns["modality"]
mod2 = input_train_mod2.uns["modality"]
dataset_id = input_train_mod1.uns["dataset_id"]
print(f"Modalities: {mod1} -> {mod2}", flush=True)

# 'cite' when ADT is involved (the CITE model expects protein targets or inputs), 'multi' for ATAC/GEX
task_type = "cite" if "ADT" in (mod1, mod2) else "multi"
print(f"Task type: {task_type}", flush=True)

train_inputs = to_sparse_csr(input_train_mod1.layers["normalized"]).astype(np.float32)
train_targets = to_sparse_csr(input_train_mod2.layers["normalized"]).astype(np.float32)
train_batches = input_train_mod1.obs["batch"].astype(str).values
test_inputs = to_sparse_csr(input_test_mod1.layers["normalized"]).astype(np.float32) if input_test_mod1 is not None else None
test_batches = input_test_mod1.obs["batch"].astype(str).values if input_test_mod1 is not None else None
mod1_var_names = input_train_mod1.var_names.to_numpy()
mod2_var = input_train_mod2.var.copy()
del input_train_mod1, input_train_mod2, input_test_mod1
gc.collect()

# ---- Cells without target signal ----
# The loss is a per-cell correlation and the targets are row-normalized: both are undefined for a constant target
# vector (e.g. a cell without any protein counts), and a single such cell turns the whole training into NaN.
keep = informative_cells(train_targets)
if not keep.all():
    print(f"Dropping {(~keep).sum()} training cells with a constant target vector", flush=True)
    train_inputs, train_targets, train_batches = train_inputs[keep], train_targets[keep], train_batches[keep]

# ---- Metadata: the derived files of the original pipeline, rebuilt from the h5ad files ----
print("Computing cell statistics...", flush=True)
train_cell_statistics = compute_cell_statistics(train_inputs, task_type)
test_cell_statistics = compute_cell_statistics(test_inputs, task_type) if test_inputs is not None else None
# the original standardized the statistics over train and test cells together
all_cell_statistics = pd.concat([train_cell_statistics, test_cell_statistics], ignore_index=True)
cell_statistics_standardization = fit_standardization(all_cell_statistics)
train_cell_statistics = apply_standardization(train_cell_statistics, cell_statistics_standardization)

batch_singular_vectors = None
if task_type == "cite":
    print("Computing batch singular vectors...", flush=True)
    train_batch_medians = compute_batch_input_medians(train_inputs, train_batches)
    batch_singular_vectors = BatchSingularVectors().fit(train_batch_medians)
    if test_inputs is not None:
        # batches of the test set that were not seen in training are projected with the fitted components; for
        # batches present in both, the training cells (many more of them) define the statistics
        new_batches = np.setdiff1d(np.unique(test_batches), train_batch_medians.index)
        if len(new_batches):
            in_new_batch = np.isin(test_batches, new_batches)
            test_batch_medians = compute_batch_input_medians(test_inputs[in_new_batch], test_batches[in_new_batch])
            batch_singular_vectors.table = pd.concat([batch_singular_vectors.table, batch_singular_vectors.transform(test_batch_medians)])

metadata_kwargs = {"task_type": task_type, "day_pattern": par["day_pattern"], "donor_pattern": par["donor_pattern"]}
train_metadata = build_metadata(
    train_batches,
    train_cell_statistics,
    batch_sv=batch_singular_vectors.lookup(train_batches) if batch_singular_vectors is not None else None,
    **metadata_kwargs,
)
test_metadata = None
if test_inputs is not None:
    test_metadata = build_metadata(
        test_batches,
        apply_standardization(test_cell_statistics, cell_statistics_standardization),
        batch_sv=batch_singular_vectors.lookup(test_batches) if batch_singular_vectors is not None else None,
        **metadata_kwargs,
    )

# ---- CITE input masks: the genes whose raw expression is appended to the SVD components ----
data_dir = tempfile.mkdtemp()
if task_type == "cite":
    print("Building the CITE input gene masks...", flush=True)
    hgnc_path, reactome_path = par.get("hgnc_complete_set"), par.get("reactome_pathways")
    if not (hgnc_path and reactome_path):
        # the docker image ships the files in SS_OPM_REFERENCE_DIR; download them when running elsewhere
        hgnc_path, reactome_path = download_reference_files(os.environ.get("SS_OPM_REFERENCE_DIR", os.path.join(data_dir, "reference")))
    hgnc = read_hgnc(hgnc_path)
    if mod1 == "GEX":
        gene_symbols = gene_symbols_from_var_names(mod1_var_names, hgnc)
        protein_symbols = gene_symbols_from_var_names(mod2_var.index.to_numpy(), hgnc)
        targets_gene2idx = make_targets_gene2idx(protein_symbols, hgnc)
        inputs_lognorm = to_sparse_csr(
            np.vstack(
                [median_normalized_log_expression(train_inputs[start : start + ROW_BLOCK]) for start in range(0, train_inputs.shape[0], ROW_BLOCK)]
            )
        )
        pair_mask, pathway_mask = make_cite_input_masks(
            inputs_lognorm,
            row_normalize(to_dense(train_targets)),
            gene_symbols,
            targets_gene2idx,
            train_batches,
            read_reactome_gmt(reactome_path),
        )
        del inputs_lognorm
        print(f"{pair_mask.any(axis=1).sum()} genes paired with a protein, {pathway_mask.sum()} pathway genes selected", flush=True)
    else:
        # ADT -> GEX: the inputs are proteins, there is nothing to pair them with; keep the SVD components only
        pair_mask = np.zeros((train_inputs.shape[1], train_targets.shape[1]), dtype=bool)
        pathway_mask = np.zeros(train_inputs.shape[1], dtype=bool)
    np.savez(os.path.join(data_dir, "cite_inputs_targets_pair3g.npz"), mask=pair_mask)
    np.savez(os.path.join(data_dir, "cite_inputs_mask2.npz"), mask=pathway_mask)
    gc.collect()

# ---- Parameters: the authors' defaults ----
pre_post_process_params = PrePostProcessing.get_params(task_type=task_type, data_dir=data_dir, device=device, seed=SEED)
# The original fits the input SVD on the training and test inputs together (transductive). The test set is an optional
# input of this component: without it, the original's own switch makes the SVD fit on the training cells only.
pre_post_process_params["use_test_inputs"] = test_inputs is not None
model_params = EncoderDecoder.get_params(task_type=task_type, device=device)
model_params["epoch"] = par["n_epochs"]
model_params["burnin_length_epoch"] = par["burnin_length_epoch"]

# ---- Fit preprocessing ----
print("Fitting preprocessing...", flush=True)
pre_post_process = PrePostProcessing(pre_post_process_params)
pre_post_process.fit_preprocess(
    inputs_values=train_inputs,
    targets_values=train_targets,
    metadata=train_metadata,
    test_inputs_values=test_inputs,
    test_metadata=test_metadata,
)

print("Preprocessing training data...", flush=True)
preprocessed_inputs, preprocessed_targets = pre_post_process.preprocess(
    inputs_values=train_inputs, targets_values=train_targets, metadata=train_metadata
)
preprocessed_inputs = np.asarray(preprocessed_inputs, dtype=np.float32)
preprocessed_targets = np.asarray(preprocessed_targets, dtype=np.float32)
print(f"model input shape X:{preprocessed_inputs.shape} Y:{preprocessed_targets.shape}", flush=True)
gc.collect()

# ---- Train ----
print("Training model...", flush=True)
model = EncoderDecoder(model_params)
model.fit(
    x=train_inputs,
    preprocessed_x=preprocessed_inputs,
    y=train_targets,
    preprocessed_y=preprocessed_targets,
    metadata=train_metadata,
    pre_post_process=pre_post_process,
)
gc.collect()

# ---- Prediction scale ----
# The network outputs per-cell z-scores (the competition only scored per-cell correlations). Fit one global affine
# map from those to the normalized targets on a subsample of training cells; it leaves every correlation unchanged.
rng = np.random.default_rng(SEED)
n_rescaling_cells = min(par["n_rescaling_cells"], train_inputs.shape[0])
rescaling_cells = np.sort(rng.choice(train_inputs.shape[0], size=n_rescaling_cells, replace=False))
train_predictions = model.predict(
    x=train_inputs[rescaling_cells],
    preprocessed_x=preprocessed_inputs[rescaling_cells],
    metadata=train_metadata.iloc[rescaling_cells].reset_index(drop=True),
)
rescaling = fit_prediction_rescaling(train_predictions, train_targets[rescaling_cells])
print(f"Prediction rescaling: slope {rescaling['slope']:.4f}, intercept {rescaling['intercept']:.4f}", flush=True)

# ---- Save ----
print("Saving model...", flush=True)
os.makedirs(par["output"], exist_ok=True)
model_dir = os.path.join(par["output"], "model")
os.makedirs(model_dir, exist_ok=True)
model.save(model_dir)
with open(os.path.join(par["output"], "pre_post_process.pickle"), "wb") as handle:
    pickle.dump(pre_post_process, handle)
with open(os.path.join(par["output"], "batch_singular_vectors.pickle"), "wb") as handle:
    pickle.dump(batch_singular_vectors, handle)
mod2_var.to_parquet(os.path.join(par["output"], "mod2_var.parquet"))
save_json(
    os.path.join(par["output"], "task_info.json"),
    {
        "task_type": task_type,
        "mod1": mod1,
        "mod2": mod2,
        "dataset_id": dataset_id,
        "day_pattern": par["day_pattern"],
        "donor_pattern": par["donor_pattern"],
        "cell_statistics_standardization": cell_statistics_standardization,
        "prediction_rescaling": rescaling,
    },
)
print("Done!", flush=True)
