import logging
import os
import tempfile

import anndata as ad
import numpy as np
import torch
from scipy.sparse import csc_matrix, issparse

import sciPENN.sciPENN_API as sciPENN_api_module
from sciPENN.sciPENN_API import sciPENN_API

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_train_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod2.h5ad",
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/test_mod1.h5ad",
    "output": "output.h5ad",
    "n_epochs": 10000,
    "batch_size": 128,
    "seed": 0,
}
meta = {"name": "scipenn"}
## VIASH END

logger.info("Reading input files...")
rna_train = ad.read_h5ad(par["input_train_mod1"])   # mod1 = RNA (GEX)
prot_train = ad.read_h5ad(par["input_train_mod2"])   # mod2 = protein (ADT)
rna_test = ad.read_h5ad(par["input_test_mod1"])

# sciPENN only supports the GEX -> ADT direction. It biologically interprets the
# gene sets as RNA and the protein sets as surface protein, so running it on the
# swapped (ADT -> GEX) direction produces meaningless output. Fail loudly instead.
mod1 = rna_train.uns.get("modality")
mod2 = prot_train.uns.get("modality")
if mod1 != "GEX" or mod2 != "ADT":
    raise ValueError(
        f"sciPENN only supports predicting protein (ADT) from RNA (GEX); "
        f"got mod1={mod1!r}, mod2={mod2!r}."
    )

np.random.seed(par["seed"])
torch.manual_seed(par["seed"])

# Feed sciPENN the task's log_cp10k layer (its internal normalize_total/log1p are
# disabled below) so predictions live in the same space as the ground truth. sciPENN
# never casts dtypes, and float64 input crashes its float32 BatchNorm layers.
for adata in (rna_train, prot_train, rna_test):
    adata.X = adata.layers["normalized"].astype(np.float32)

# sciPENN scales each batch independently with sc.pp.scale, which divides by (n - 1)
# and therefore fails on any batch with a single cell. Only use per-batch scaling when
# every non-empty batch in both train and test has at least 2 cells; otherwise fall
# back to global scaling (batch keys disabled).
def _batches_safe(*adatas):
    for adata in adatas:
        if "batch" not in adata.obs:
            return False
        counts = adata.obs["batch"].value_counts()
        if (counts[counts > 0] < 2).any():
            return False
    return True

if _batches_safe(rna_train, rna_test):
    train_batchkeys = ["batch"]
    test_batchkey = "batch"
else:
    logger.warning(
        "Disabling per-batch scaling: a batch has fewer than 2 cells "
        "(sciPENN's per-batch sc.pp.scale would divide by zero)."
    )
    train_batchkeys = None
    test_batchkey = None

# sciPENN z-scores the protein targets (per batch or globally, as chosen above) with
# sc.pp.scale, so the model predicts z-scores. Record the training-protein mean/std over
# the same groups to map predictions back to log_cp10k. With global scaling this is the
# exact inverse; with per-batch scaling the test batches are unseen, so use the
# cell-weighted average of the per-batch statistics. Must run before sciPENN_API, which
# scales prot_train.X in place.
def _scale_stats(X):
    # Matches sc.pp.scale: ddof=1 std, zero std replaced by 1.
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return X.mean(axis=0), std

prot_X = prot_train.X.toarray() if issparse(prot_train.X) else np.asarray(prot_train.X)
prot_X = prot_X.astype(np.float64)
if train_batchkeys is None:
    prot_mean, prot_std = _scale_stats(prot_X)
else:
    batches = prot_train.obs["batch"].to_numpy()
    group_masks = [batches == batch_label for batch_label in np.unique(batches)]
    stats = [_scale_stats(prot_X[mask]) for mask in group_masks]
    weights = np.array([mask.sum() for mask in group_masks], dtype=np.float64)
    weights /= weights.sum()
    prot_mean = sum(weight * mean for weight, (mean, _) in zip(weights, stats))
    prot_std = sum(weight * std for weight, (_, std) in zip(weights, stats))
del prot_X

# sciPENN 0.9.6's build_dir loops forever on absolute paths (os.path.split("/") never
# yields ""), appending "/" to a list until the process is OOM-killed.
sciPENN_api_module.build_dir = lambda path: os.makedirs(path, exist_ok=True)

# sciPENN auto-falls back to CPU, but pass the detected value explicitly.
use_gpu = torch.cuda.is_available()
logger.info("Using %s", "GPU" if use_gpu else "CPU")

logger.info("Constructing sciPENN model...")
weights_dir = tempfile.mkdtemp(prefix="scipenn_")
scipenn = sciPENN_API(
    gene_trainsets=[rna_train],
    protein_trainsets=[prot_train],
    gene_test=rna_test,
    train_batchkeys=train_batchkeys,
    test_batchkey=test_batchkey,
    # HVG selection and QC filtering are data preprocessing, not model hyperparameters,
    # so they are not exposed as arguments. Keep sciPENN's HVG selection on, and disable
    # its cell/gene QC filtering so the prediction keeps every test cell (see the row-count
    # check below).
    select_hvg=True,
    cell_normalize=False,
    log_normalize=False,
    min_cells=0,
    min_genes=0,
    batch_size=par["batch_size"],
    use_gpu=use_gpu,
)

logger.info("Training sciPENN...")
scipenn.train(n_epochs=par["n_epochs"], weights_dir=weights_dir, load=False)

logger.info("Predicting protein expression...")
imputed = scipenn.predict()   # .X = predicted protein z-scores

# Row space must equal the full test set, in the original order. With
# min_genes=min_cells=0 no cells are dropped, so verify the assumption holds.
if imputed.n_obs != rna_test.n_obs:
    raise RuntimeError(
        f"sciPENN returned {imputed.n_obs} cells but the test set has "
        f"{rna_test.n_obs}; QC filtering must be disabled (min_genes/min_cells=0)."
    )
if not np.array_equal(np.asarray(imputed.obs_names), np.asarray(rna_test.obs_names)):
    imputed = imputed[rna_test.obs_names].copy()

# Column space must equal input_train_mod2.var order. sciPENN may reorder/subset
# proteins, so scatter by name and zero-fill any protein it dropped (NaN would break
# the correlation/mse metrics).
imputed_X = np.asarray(imputed.X, dtype=np.float64) * prot_std + prot_mean
src_pos = {name: i for i, name in enumerate(imputed.var_names)}
target = list(prot_train.var_names)
preds = np.zeros((imputed.n_obs, len(target)), dtype=np.float32)
missing = []
for col_idx, name in enumerate(target):
    src_idx = src_pos.get(name)
    if src_idx is not None:
        preds[:, col_idx] = imputed_X[:, src_idx]
    else:
        missing.append(name)
if missing:
    logger.warning("sciPENN dropped %d proteins; zero-filled: %s", len(missing), missing)

logger.info("Writing predictions...")
out = ad.AnnData(
    layers={"normalized": csc_matrix(preds)},
    obs=rna_test.obs[[]],
    var=prot_train.var[[]],
    uns={
        "dataset_id": rna_test.uns["dataset_id"],
        "method_id": meta["name"],
    },
)
out.write_h5ad(par["output"], compression="gzip")
logger.info("Predictions saved to %s", par["output"])
