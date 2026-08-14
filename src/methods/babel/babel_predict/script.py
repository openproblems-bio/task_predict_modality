import logging
import pickle
import sys

import anndata as ad
import numpy as np
import torch
from scipy.sparse import csc_matrix, issparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_test_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/test_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/train_mod2.h5ad",
    "input_model": "output_model.pkl",
    "output": "output_pred.h5ad",
}
meta = {"name": "babel", "resources_dir": "src/methods/babel"}
## VIASH END

sys.path.append(meta["resources_dir"])

from model import AssymSplicedAutoEncoder


def _to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def _lognorm_per_cell(pred_counts, target_sum=1e4):
    """Convert the NB decoder's raw-count-scale mean prediction into the same
    log1p(normalized-to-target_sum) space used for the "normalized" layer
    elsewhere in this codebase (see babel_train._rna_matrix / senkin_tmp's
    senkin_tmp_cite_pred.preprocess.log_normalize), since file_prediction.yaml
    requires "normalized" to hold log-normalized values, not raw NB means
    (which are unbounded and on an arbitrary count scale)."""
    pred_counts = np.clip(pred_counts, a_min=0, a_max=None)
    row_sums = pred_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return np.log1p(pred_counts / row_sums * target_sum)


logger.info("Reading input files...")
adata_test_mod1 = ad.read_h5ad(par["input_test_mod1"])
adata_train_mod2 = ad.read_h5ad(par["input_train_mod2"])

logger.info("Loading model bundle...")
with open(par["input_model"], "rb") as f:
    bundle = pickle.load(f)

test_modality = adata_test_mod1.uns.get("modality")
if test_modality != "ATAC":
    raise ValueError(
        f"babel_predict only supports ATAC->GEX prediction (test input must be ATAC), "
        f"got modality={test_modality!r}. The GEX->ATAC direction of this BABEL port "
        "collapses to a per-peak base-rate prediction that ignores the RNA input "
        "(confirmed via cross-cell prediction similarity ~0.6 and near-zero separation "
        "between accessible/inaccessible peaks, even after training to convergence) "
        "and is intentionally disabled rather than silently returning uninformative output."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AssymSplicedAutoEncoder(bundle["n_genes"], bundle["chrom_counts"], hidden_dim=bundle["hidden_dim"])
model.load_state_dict(bundle["state_dict"])
model.to(device)
model.eval()

chrom_groups = bundle["chrom_groups"]

if list(adata_test_mod1.var_names) != bundle["atac_var_names"]:
    raise ValueError(
        "Test ATAC var_names do not match the peak order the model was trained with; "
        "reindexing across mismatched peak sets is not supported."
    )
X_bin = (_to_dense(adata_test_mod1.layers.get("counts", adata_test_mod1.X)) > 0).astype(np.float32)
X_per_chrom = [torch.from_numpy(X_bin[:, idxs]).to(device) for idxs in chrom_groups.values()]
with torch.no_grad():
    encoded = model.encoder2(X_per_chrom)
    pred_mean, _, _ = model.decoder1(encoded)
pred = _lognorm_per_cell(pred_mean.cpu().numpy())
out_var = adata_train_mod2.var

logger.info("Writing predictions...")
adata_out = ad.AnnData(
    layers={"normalized": csc_matrix(pred)},
    obs=adata_test_mod1.obs,
    var=out_var,
    uns={
        "dataset_id": adata_test_mod1.uns.get("dataset_id", ""),
        "method_id": "babel",
    },
)

adata_out.write_h5ad(par["output"], compression="gzip")
logger.info("Predictions saved to %s", par["output"])
