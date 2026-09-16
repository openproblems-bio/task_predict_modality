import gc
import logging
import pickle
import sys

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import KFold

from senkin_tmp_cite_pred.preprocess import (
    clr_tsvd,
    get_top_correlated_features,
    log_normalize,
    remove_constant_vars,
    senkin_normalize,
    to_dense,
)
from senkin_tmp_cite_pred.lgbm_models import get_lgbm_predictions, lgbm_params_1, lgbm_params_2, lgbm_params_3, lgbm_params_4
from senkin_tmp_cite_pred.nn_models import cite_cos_sim_model, cite_mse_model, nn_kfold, prepare_nn_inputs, zscore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## VIASH START
par = {
    "input_train_mod1": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod1.h5ad",
    "input_train_mod2": "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/train_mod2.h5ad",
    "input_test_mod1":  "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/test_mod1.h5ad",
    "output": "output_model.pkl",
    "n_folds": 5,
    "lgbm_boost_rounds": 10000,
    "lgbm_early_stopping": 100,
    "nn_epochs": 100,
    "n_tsvd_components": 100,
}
meta = {"name": "senkin_tmp", "resources_dir": "src/methods/senkin_tmp/senkin_tmp_train", "cpus": None}
## VIASH END

sys.path.append(meta["resources_dir"])
from exit_codes import exit_non_applicable

# The original solution corrected batch effects per day and computed gene-protein correlations per donor and day.
# The benchmark datasets only carry a generic `batch` column (e.g. "s1d1" in NeurIPS 2021, "day_donor" in
# NeurIPS 2022), which is used for both purposes here. Nothing dataset specific is assumed about its format.
BATCH_KEY = "batch"
SEED = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
logger.info("Reading input files...")
adata_rna_train = ad.read_h5ad(par["input_train_mod1"])
adata_prot_train = ad.read_h5ad(par["input_train_mod2"])
adata_rna_test = ad.read_h5ad(par["input_test_mod1"])

# senkin is a CITE-seq method that predicts protein (ADT) from RNA (GEX); it treats
# mod1 as RNA and mod2 as protein. Skip the datasets/directions it cannot handle
# (e.g. Multiome, or the ADT->GEX swap) instead of running for hours and OOMing.
_mod1 = adata_rna_train.uns.get("modality")
_mod2 = adata_prot_train.uns.get("modality")
if _mod1 != "GEX" or _mod2 != "ADT":
    exit_non_applicable(
        f"senkin only supports predicting protein (ADT) from RNA (GEX); "
        f"got mod1={_mod1!r}, mod2={_mod2!r}."
    )

# Align protein cells with RNA cells
adata_prot_train = adata_prot_train[adata_rna_train.obs_names].copy()

for adata in (adata_rna_train, adata_rna_test):
    if BATCH_KEY not in adata.obs.columns:
        adata.obs[BATCH_KEY] = "all"
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype(str)
adata_prot_train.obs[BATCH_KEY] = adata_rna_train.obs[BATCH_KEY].values

# Concatenate train + test RNA so the original pipeline sees both at once
# (all unsupervised transformations were fit on train and test cells together)
adata_rna_train.obs["split"] = "train"
adata_rna_test.obs["split"] = "test"
adata_rna_all = ad.concat([adata_rna_train, adata_rna_test], axis=0, join="inner", merge="same")
adata_rna_all.X = adata_rna_all.layers["counts"]
adata_rna_all.obs[BATCH_KEY] = adata_rna_all.obs[BATCH_KEY].astype(str)
del adata_rna_all.layers["counts"]
if "normalized" in adata_rna_all.layers:
    del adata_rna_all.layers["normalized"]

# ---------------------------------------------------------------------------
# Preprocessing on combined train+test RNA
# ---------------------------------------------------------------------------
logger.info("Preprocessing RNA (train + test combined)...")

adata_rna_all = remove_constant_vars(adata_rna_all)
train_idx = np.flatnonzero((adata_rna_all.obs["split"] == "train").values)
test_idx = np.flatnonzero((adata_rna_all.obs["split"] == "test").values)
n_train, n_test = len(train_idx), len(test_idx)
logger.info(f"{n_train} train cells, {n_test} test cells, {adata_rna_all.n_vars} non-constant genes")

# Log-normalize the way the competition inputs were normalized: log1p(counts per million)
X_lognorm_all = log_normalize(adata_rna_all, target_sum=1e6).tocsr()

logger.info("Computing CLR-TSVD...")
X_clr_tsvd_all = clr_tsvd(adata_rna_all, n_components=200, random_state=SEED)

# Custom sqrt normalization with per-batch median correction, reduced with TSVD (100) and PCA (64) as in the original
logger.info("Computing SenKin normalization, TSVD and PCA...")
X_sqrt_norm_all = senkin_normalize(adata_rna_all, batch_key=BATCH_KEY)
n_sqrt_tsvd = min(100, min(X_sqrt_norm_all.shape) - 1)
X_sqrt_tsvd_all = TruncatedSVD(n_components=n_sqrt_tsvd, algorithm="arpack", random_state=SEED).fit_transform(X_sqrt_norm_all)
n_sqrt_pca = min(64, min(X_sqrt_norm_all.shape) - 1)
X_sqrt_pca_all = PCA(n_components=n_sqrt_pca, copy=False, random_state=SEED).fit_transform(X_sqrt_norm_all)
del X_sqrt_norm_all
gc.collect()

# Correlated gene selection: computed on train cells only (no label leakage), on log-normalized RNA and
# normalized proteins, per batch. As in the original, the raw counts of the selected genes are used as features.
logger.info("Selecting correlated features...")
adata_rna_train_filt = adata_rna_all[train_idx].copy()
adata_rna_train_filt.obsm["X_log_normalized"] = X_lognorm_all[train_idx]
top_corr_genes = get_top_correlated_features(
    adata_rna_train_filt,
    adata_prot_train,
    group_key=BATCH_KEY,
    quantile_threshold=0.1,
    top_n=10,
    rna_key="X_log_normalized",
    prot_key="normalized",
)
del adata_rna_train_filt

# The original solution additionally used a hand-curated list of genes encoding the measured proteins.
# Generic equivalent: genes whose name matches a protein name.
def _feature_names(adata):
    names = adata.var["feature_name"] if "feature_name" in adata.var.columns else adata.var_names.to_series()
    return names.astype(str).str.upper()

_gene_names = _feature_names(adata_rna_all)
_protein_names = set(_feature_names(adata_prot_train))
known_genes = adata_rna_all.var_names[_gene_names.isin(_protein_names).values].tolist()
selected_genes = sorted(set(top_corr_genes) | set(known_genes))
logger.info(f"{len(top_corr_genes)} correlated genes + {len(known_genes)} protein-encoding genes = {len(selected_genes)} selected genes")
X_raw_selected_all = to_dense(adata_rna_all[:, selected_genes].X, dtype=np.float32)

X_counts_all = adata_rna_all.X.tocsr()

# Protein targets
Y_prot_train = to_dense(adata_prot_train.layers["normalized"], dtype=np.float64)
Y_prot_raw = to_dense(adata_prot_train.layers.get("counts", adata_prot_train.X), dtype=np.float64)

folds = KFold(n_splits=par["n_folds"], shuffle=True, random_state=666)
n_tsvd = par["n_tsvd_components"]
boost_rounds = par["lgbm_boost_rounds"]
early_stop = par["lgbm_early_stopping"]

# Pin LightGBM to the allocated cores. Its default (num_threads=0) spawns one thread
# per core the container *sees* (the whole node) while the job is cgroup-throttled to
# meta["cpus"], so the threads oversubscribe and thrash -- the same class of slowdown
# fixed for guanlab in #59. Leave the library default when cpus is unknown (local runs).
_n_threads = meta.get("cpus")
if _n_threads:
    for _p in (lgbm_params_1, lgbm_params_2, lgbm_params_3, lgbm_params_4):
        _p["num_threads"] = _n_threads

# ---------------------------------------------------------------------------
# LightGBM — 4 models, train+test passed together (original design)
# get_lgbm_predictions concatenates train+test, fits TSVD on combined array
# ---------------------------------------------------------------------------
def _lgbm(X_all, Y, params, description):
    logger.info(f"Training LightGBM {description}...")
    return get_lgbm_predictions(
        X_all[train_idx], Y, X_all[test_idx], folds, params,
        n_tsvd_components=n_tsvd, num_boost_round=boost_rounds, early_stopping_rounds=early_stop,
    )

lgbm1_svd_all = _lgbm(X_lognorm_all, Y_prot_train, lgbm_params_1, "model 1 (log-normalized RNA -> proteins)")

X_comb_all = np.concatenate([X_clr_tsvd_all, X_raw_selected_all, X_sqrt_tsvd_all, X_sqrt_pca_all], axis=1)
lgbm2_svd_all = _lgbm(X_comb_all, Y_prot_train, lgbm_params_2, "model 2 (CLR-TSVD + selected genes + normalized TSVD/PCA -> proteins)")
del X_comb_all

lgbm3_svd_all = _lgbm(X_counts_all, Y_prot_train, lgbm_params_3, "model 3 (raw counts -> proteins)")
lgbm4_svd_all = _lgbm(X_counts_all, Y_prot_raw, lgbm_params_4, "model 4 (raw counts -> raw proteins)")
del X_counts_all, X_lognorm_all
gc.collect()

# ---------------------------------------------------------------------------
# Neural networks. Every feature block is z-scored per cell before concatenation, as in the original.
# ---------------------------------------------------------------------------
def _nn_inputs(idx):
    return prepare_nn_inputs(
        X_clr_tsvd_all[idx], X_raw_selected_all[idx], X_sqrt_tsvd_all[idx], X_sqrt_pca_all[idx],
        lgbm1_svd_all[idx], lgbm2_svd_all[idx], lgbm3_svd_all[idx], lgbm4_svd_all[idx],
    )

# get_lgbm_predictions returns train cells first, then test cells
nn_X_train = _nn_inputs(np.arange(n_train))
nn_X_test = _nn_inputs(np.arange(n_train, n_train + n_test))
nn_y_train = Y_prot_train.astype(np.float32)

train_cell_ids = np.array(adata_rna_all.obs_names[train_idx])
test_cell_ids = np.array(adata_rna_all.obs_names[test_idx])

logger.info("Training neural network (cosine model)...")
train_preds_cos, test_preds_cos = nn_kfold(
    train_cell_ids, nn_X_train, nn_y_train,
    test_cell_ids, nn_X_test,
    cite_cos_sim_model, folds,
    model_name="cite_cos_model",
    BATCH_SIZE=620, EPOCHS=par["nn_epochs"], LR_FACTOR=0.05,
    models_dir="models",
)

logger.info("Training neural network (MSE model)...")
nn_y_train_z = zscore(nn_y_train)
train_preds_mse, test_preds_mse = nn_kfold(
    train_cell_ids, nn_X_train, nn_y_train_z,
    test_cell_ids, nn_X_test,
    cite_mse_model, folds,
    model_name="cite_mse_model",
    BATCH_SIZE=600, EPOCHS=par["nn_epochs"], LR_FACTOR=0.1,
    models_dir="models",
)

# Blend — identical to original train_nn_models
train_preds = zscore(train_preds_cos) * 0.55 + zscore(train_preds_mse) * 0.45
test_preds = zscore(test_preds_cos) * 0.55 + zscore(test_preds_mse) * 0.45

# The original solution was scored with a per-cell Pearson correlation only, so its predictions are z-scored per
# cell. The benchmark also computes RMSE/MAE, so bring the predictions back to the scale of the normalized proteins
# with a single global affine transform fitted on the out-of-fold training predictions. One (slope, intercept) pair
# for all cells and proteins leaves every per-cell and per-protein correlation untouched.
slope, intercept = np.polyfit(train_preds.ravel(), Y_prot_train.ravel(), deg=1)
logger.info(f"Rescaling z-scored predictions to the target scale: slope {slope:.4f}, intercept {intercept:.4f}")
test_preds = test_preds * slope + intercept

# ---------------------------------------------------------------------------
# Save bundle — test predictions stored directly, predict script just reads them
# ---------------------------------------------------------------------------
logger.info("Saving model bundle...")
bundle = {
    "test_predictions": test_preds.astype(np.float32),  # (n_test, n_proteins)
    "test_obs_names": test_cell_ids,
    "prot_var": adata_prot_train.var,
    "dataset_id": adata_rna_train.uns.get("dataset_id", ""),
}

with open(par["output"], "wb") as f:
    pickle.dump(bundle, f, protocol=4)

logger.info("Training complete. Model saved to %s", par["output"])
