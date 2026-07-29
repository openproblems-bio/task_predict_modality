import gc
import logging
import pickle

import anndata as ad
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import tensorflow as tf

from senkin_tmp_cite_pred.preprocess import remove_constant_vars, senkin_normalize, get_top_correlated_features, log_normalize, clr_tsvd
from senkin_tmp_cite_pred.lgbm_models import get_lgbm_predictions, lgbm_params_1, lgbm_params_2, lgbm_params_3, lgbm_params_4
from senkin_tmp_cite_pred.nn_models import cite_cos_sim_model, cite_mse_model, nn_kfold, zscore
from senkin_tmp_cite_pred.metrics import cosine_similarity_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_train_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod2.h5ad",
    "input_test_mod1":  "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad",
    "output": "output_model.pkl",
    "n_folds": 5,
    "lgbm_boost_rounds": 10000,
    "lgbm_early_stopping": 100,
    "nn_epochs": 100,
    "n_tsvd_components": 100,
}
meta = {"name": "senkin"}
## VIASH END


def _to_dense(X):
    return X.toarray() if issparse(X) else np.array(X)


def _parse_batch(obs, batch_col="batch"):
    if batch_col not in obs.columns:
        obs["day"] = "unknown"
        obs["donor"] = "unknown"
        return obs
    def _split(b):
        b = str(b)
        d_idx = b.find("d")
        if d_idx > 0:
            return b[1:d_idx], b[d_idx + 1:]
        return b, b
    obs["day"], obs["donor"] = zip(*obs[batch_col].astype(str).map(_split))
    return obs


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
logger.info("Reading input files...")
adata_rna_train  = ad.read_h5ad(par["input_train_mod1"])
adata_prot_train = ad.read_h5ad(par["input_train_mod2"])
adata_rna_test   = ad.read_h5ad(par["input_test_mod1"])

adata_rna_train.obs  = _parse_batch(adata_rna_train.obs)
adata_prot_train.obs = _parse_batch(adata_prot_train.obs)
adata_rna_test.obs   = _parse_batch(adata_rna_test.obs)

# Mark split membership before concatenating
adata_rna_train.obs["split"] = "train"
adata_rna_test.obs["split"]  = "test"

# Concatenate train + test RNA so the original pipeline sees both at once
import scanpy as sc
adata_rna_all = sc.concat([adata_rna_train, adata_rna_test], axis=0)
adata_rna_all.X = adata_rna_all.layers["counts"]

# ---------------------------------------------------------------------------
# Preprocessing on combined train+test RNA
# ---------------------------------------------------------------------------
logger.info("Preprocessing RNA (train + test combined)...")

adata_rna_all_filt = remove_constant_vars(adata_rna_all)
X_counts_all = _to_dense(adata_rna_all_filt.layers.get("counts", adata_rna_all_filt.X))

train_mask = adata_rna_all_filt.obs["split"] == "train"
test_mask  = adata_rna_all_filt.obs["split"] == "test"

# Log-normalize (CP10K + log1p) via the senkin_tmp_cite_pred library helper.
# log_normalize preserves the input sparsity, so densify for the array math below.
X_lognorm_all = _to_dense(log_normalize(adata_rna_all_filt, target_sum=1e4)).astype(np.float64)

# CLR-TSVD via the library helper (random_state=42 for reproducible components)
logger.info("Computing CLR-TSVD...")
X_clr_tsvd_all = clr_tsvd(adata_rna_all_filt, n_components=200, random_state=42)

# SenKin normalization + PCA — fit on all
logger.info("Computing SenKin normalization and PCA...")
X_sqrt_norm_all = np.asarray(senkin_normalize(adata_rna_all_filt, batch_key="day"))
n_pca = min(100, min(X_sqrt_norm_all.shape) - 1)
pca_model = PCA(n_components=n_pca, random_state=42)
X_pca_all = pca_model.fit_transform(X_sqrt_norm_all)

# Correlated gene selection — computed on train cells only (no label leakage)
logger.info("Selecting correlated features...")
adata_rna_train_filt = adata_rna_all_filt[train_mask]
Y_prot_train = _to_dense(adata_prot_train.layers.get("normalized", adata_prot_train.X)).astype(np.float64)
# get_top_correlated_features hardcodes .layers["dsb"]; alias our normalized layer
_added_dsb = "dsb" not in adata_prot_train.layers
if _added_dsb:
    adata_prot_train.layers["dsb"] = adata_prot_train.layers.get("normalized", adata_prot_train.X)
# Use "day" as group key — benchmark data has no "donor" column
_group_key = "donor" if "donor" in adata_rna_train_filt.obs.columns else "day"
top_corr_genes = get_top_correlated_features(adata_rna_train_filt, adata_prot_train, group_key=_group_key)
if _added_dsb:
    del adata_prot_train.layers["dsb"]
all_var_names = list(adata_rna_all_filt.var_names)
selected_gene_idxs = [all_var_names.index(g) for g in top_corr_genes if g in all_var_names]
X_raw_selected_all = X_counts_all[:, selected_gene_idxs].astype(np.float64)

# Split back into train / test portions
X_lognorm_train    = X_lognorm_all[train_mask]
X_lognorm_test     = X_lognorm_all[test_mask]
X_clr_tsvd_train   = X_clr_tsvd_all[train_mask]
X_clr_tsvd_test    = X_clr_tsvd_all[test_mask]
X_pca_train        = X_pca_all[train_mask]
X_pca_test         = X_pca_all[test_mask]
X_raw_sel_train    = X_raw_selected_all[train_mask]
X_raw_sel_test     = X_raw_selected_all[test_mask]
X_counts_train     = X_counts_all[train_mask]
X_counts_test      = X_counts_all[test_mask]

folds = KFold(n_splits=par["n_folds"], shuffle=True, random_state=666)
n_tsvd       = par["n_tsvd_components"]
boost_rounds = par["lgbm_boost_rounds"]
early_stop   = par["lgbm_early_stopping"]

# Protein targets
Y_prot_raw = _to_dense(adata_prot_train.layers.get("counts", adata_prot_train.X)).astype(np.float64)

# ---------------------------------------------------------------------------
# LightGBM — 4 models, train+test passed together (original design)
# get_lgbm_predictions concatenates train+test, fits TSVD on combined array
# ---------------------------------------------------------------------------
logger.info("Training LightGBM model 1 (log-norm → proteins)...")
lgbm1_svd_all = get_lgbm_predictions(
    X_lognorm_train, Y_prot_train, X_lognorm_test,
    folds, lgbm_params_1,
    n_tsvd_components=n_tsvd,
    num_boost_round=boost_rounds,
    early_stopping_rounds=early_stop,
)

logger.info("Training LightGBM model 2 (combined → proteins)...")
X_comb_train = np.concatenate([X_clr_tsvd_train, X_raw_sel_train, X_pca_train], axis=1)
X_comb_test  = np.concatenate([X_clr_tsvd_test,  X_raw_sel_test,  X_pca_test],  axis=1)
lgbm2_svd_all = get_lgbm_predictions(
    X_comb_train, Y_prot_train, X_comb_test,
    folds, lgbm_params_2,
    n_tsvd_components=n_tsvd,
    num_boost_round=boost_rounds,
    early_stopping_rounds=early_stop,
)

logger.info("Training LightGBM model 3 (raw counts → proteins)...")
lgbm3_svd_all = get_lgbm_predictions(
    X_counts_train, Y_prot_train, X_counts_test,
    folds, lgbm_params_3,
    n_tsvd_components=n_tsvd,
    num_boost_round=boost_rounds,
    early_stopping_rounds=early_stop,
)

logger.info("Training LightGBM model 4 (raw counts → raw proteins)...")
lgbm4_svd_all = get_lgbm_predictions(
    X_counts_train, Y_prot_raw, X_counts_test,
    folds, lgbm_params_4,
    n_tsvd_components=n_tsvd,
    num_boost_round=boost_rounds,
    early_stopping_rounds=early_stop,
)

# get_lgbm_predictions returns shape (n_train + n_test, n_tsvd)
n_train = X_lognorm_train.shape[0]
n_test  = X_lognorm_test.shape[0]

lgbm1_tr = lgbm1_svd_all[:n_train]; lgbm1_te = lgbm1_svd_all[n_train:]
lgbm2_tr = lgbm2_svd_all[:n_train]; lgbm2_te = lgbm2_svd_all[n_train:]
lgbm3_tr = lgbm3_svd_all[:n_train]; lgbm3_te = lgbm3_svd_all[n_train:]
lgbm4_tr = lgbm4_svd_all[:n_train]; lgbm4_te = lgbm4_svd_all[n_train:]

# Build NN inputs
nn_X_train = np.concatenate([X_clr_tsvd_train, X_pca_train, X_raw_sel_train, lgbm1_tr, lgbm2_tr, lgbm3_tr, lgbm4_tr], axis=1).astype(np.float32)
nn_X_test  = np.concatenate([X_clr_tsvd_test,  X_pca_test,  X_raw_sel_test,  lgbm1_te, lgbm2_te, lgbm3_te, lgbm4_te], axis=1).astype(np.float32)
nn_y_train = Y_prot_train.astype(np.float32)

train_cell_ids = np.array(adata_rna_all_filt.obs_names[train_mask])
test_cell_ids  = np.array(adata_rna_all_filt.obs_names[test_mask])

# ---------------------------------------------------------------------------
# Neural network — use original nn_kfold which saves checkpoint weights
# ---------------------------------------------------------------------------
logger.info("Training neural network (cosine model)...")
import os
os.makedirs("models", exist_ok=True)

train_preds_cos, test_preds_cos = nn_kfold(
    train_cell_ids, nn_X_train, nn_y_train,
    test_cell_ids,  nn_X_test,
    cite_cos_sim_model, folds,
    model_name="cite_cos_model",
    BATCH_SIZE=620, EPOCHS=par["nn_epochs"], LR_FACTOR=0.05,
)

logger.info("Training neural network (MSE model)...")
nn_y_train_z = zscore(nn_y_train)
train_preds_mse, test_preds_mse = nn_kfold(
    train_cell_ids, nn_X_train, nn_y_train_z,
    test_cell_ids,  nn_X_test,
    cite_mse_model, folds,
    model_name="cite_mse_model",
    BATCH_SIZE=600, EPOCHS=par["nn_epochs"], LR_FACTOR=0.1,
)

# Blend — identical to original train_nn_models
test_preds = zscore(test_preds_cos) * 0.55 + zscore(test_preds_mse) * 0.45

# ---------------------------------------------------------------------------
# Save bundle — test predictions stored directly, predict script just reads them
# ---------------------------------------------------------------------------
logger.info("Saving model bundle...")
bundle = {
    "test_predictions": test_preds.astype(np.float32),  # (n_test, n_proteins)
    "test_obs": adata_rna_test.obs,
    "prot_var": adata_prot_train.var,
    "dataset_id": adata_rna_train.uns.get("dataset_id", ""),
}

with open(par["output"], "wb") as f:
    pickle.dump(bundle, f, protocol=4)

logger.info("Training complete. Model saved to %s", par["output"])
