import anndata as ad
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

## VIASH START
par = {
  "input_test_mod2" : "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod2.h5ad",
  "input_prediction" : "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/prediction.h5ad",
  "output" : "output/scores.h5ad",
  "leiden_resolution" : 1,
  "min_group_size" : 30,
  "padj_threshold" : 0.05,
  "effect_threshold_gex" : 0.25,
  "effect_threshold_adt" : 0.5,
  "effect_threshold_atac" : 0,
  "seed" : 0
}
## VIASH END

logging.basicConfig(level=logging.INFO)


def dense(X):
  return np.asarray(X.toarray() if sparse.issparse(X) else X, dtype=np.float64)


def leiden_groups(X, var_mask, n_comps):
  """Leiden clusters of the ground truth, as integer codes per cell."""
  tmp = ad.AnnData(X[:, var_mask])
  n_comps = min(n_comps, tmp.n_obs - 1, tmp.n_vars - 1)
  sc.pp.pca(tmp, n_comps=n_comps, random_state=par["seed"])
  sc.pp.neighbors(tmp, use_rep="X_pca", n_neighbors=15, random_state=par["seed"])
  sc.tl.leiden(tmp, resolution=par["leiden_resolution"], key_added="leiden", flavor="igraph",
               n_iterations=2, directed=False, random_state=par["seed"])
  return tmp.obs["leiden"].astype(int).to_numpy()


def de_one_vs_rest(X, codes, n_groups):
  """One-vs-rest Wilcoxon rank-sum test (tie-corrected) for every group and feature.
  Returns the signed z-score, the effect (mean of the group minus mean of the rest) and the BH-adjusted p-values,
  all groups x features."""
  tmp = ad.AnnData(X, var=pd.DataFrame(index=np.arange(X.shape[1]).astype(str)))
  tmp.obs["group"] = pd.Categorical(codes.astype(str), categories=np.arange(n_groups).astype(str))
  sc.tl.rank_genes_groups(tmp, "group", method="wilcoxon", reference="rest", tie_correct=True,
                          corr_method="benjamini-hochberg", n_genes=X.shape[1])
  res = tmp.uns["rank_genes_groups"]
  z = np.empty((n_groups, X.shape[1]))
  padj = np.empty((n_groups, X.shape[1]))
  for g in range(n_groups):
    # scanpy sorts the features by score, so put them back in feature order
    order = res["names"][str(g)].astype(int)
    z[g, order] = res["scores"][str(g)]
    padj[g, order] = res["pvals_adj"][str(g)]
  # scanpy's logfoldchanges assume log1p data, so the effect is the difference of means on the layer itself
  n_g = np.bincount(codes, minlength=n_groups)[:, None]
  sum_g = np.vstack([X[codes == g].sum(axis=0) for g in range(n_groups)])
  effect = sum_g / n_g - (X.sum(axis=0) - sum_g) / (X.shape[0] - n_g)
  return z, effect, padj


logging.info("Reading solution file")
ad_sol = ad.read_h5ad(par["input_test_mod2"])

logging.info("Reading prediction file")
ad_pred = ad.read_h5ad(par["input_prediction"])

logging.info("Check prediction format")
if ad_sol.uns["dataset_id"] != ad_pred.uns["dataset_id"]:
  raise ValueError("Prediction and solution have differing dataset_ids")

if ad_sol.shape != ad_pred.shape:
  raise ValueError("Dataset and prediction anndata objects should have the same shape / dimensions.")

# the clusters are defined on the solution and applied to the prediction, so the cells must be paired
if not (ad_pred.obs_names.equals(ad_sol.obs_names) and ad_pred.var_names.equals(ad_sol.var_names)):
  logging.info("Reordering the prediction to the cells and features of the solution")
  ad_pred = ad_pred[ad_sol.obs_names, ad_sol.var_names]

sol = dense(ad_sol.layers["normalized"])
pred = dense(ad_pred.layers["normalized"])

# score non-finite predictions as zero, rather than returning a NaN metric
non_finite = ~np.isfinite(pred)
if non_finite.any():
  logging.info("Prediction contains %d non-finite values, scoring them as 0", non_finite.sum())
  pred[non_finite] = 0

modality = ad_sol.uns["modality"]
effect_threshold = {
  "GEX": par["effect_threshold_gex"],
  "ADT": par["effect_threshold_adt"],
  "ATAC": par["effect_threshold_atac"],
}[modality]

logging.info("Clustering the solution")
# proteins: all of the few features; GEX and ATAC: the highly variable features
if modality == "ADT" or ad_sol.var["hvg"].sum() < 50:
  var_mask, n_comps = np.ones(ad_sol.n_vars, dtype=bool), 30
else:
  var_mask, n_comps = ad_sol.var["hvg"].to_numpy(dtype=bool), 50
labels = leiden_groups(sol, var_mask, n_comps)

# clusters that are too small to test are left out, together with their cells
sizes = np.bincount(labels)
groups = np.flatnonzero(sizes >= par["min_group_size"])
logging.info("%d clusters, %d with at least %d cells", len(sizes), len(groups), par["min_group_size"])
if len(groups) < 2:
  raise ValueError(f"DEG recall needs at least 2 clusters with {par['min_group_size']} cells or more, found {len(groups)}")

keep = np.isin(labels, groups)
codes = pd.Categorical(labels[keep], categories=groups).codes

logging.info("Computing DEGs of the solution and the prediction")
z_sol, effect_sol, padj_sol = de_one_vs_rest(sol[keep], codes, len(groups))
z_pred, effect_pred, padj_pred = de_one_vs_rest(pred[keep], codes, len(groups))

deg_sol = (padj_sol < par["padj_threshold"]) & (effect_sol > effect_threshold)
deg_pred = (padj_pred < par["padj_threshold"]) & (effect_pred > effect_threshold)

logging.info("Computing DEG F1")
n_deg_sol = deg_sol.sum(axis=1)
n_deg_pred = deg_pred.sum(axis=1)
n_both = (deg_sol & deg_pred).sum(axis=1)
# 2 * precision * recall / (precision + recall), which is 0 when the prediction calls no DEGs
f1_per_group = 2 * n_both[n_deg_sol > 0] / (n_deg_sol + n_deg_pred)[n_deg_sol > 0]
if f1_per_group.size:
  deg_f1 = float(f1_per_group.mean())
else:
  logging.warning("No cluster has DEGs in the solution, scoring DEG F1 as 0")
  deg_f1 = 0.0

logging.info("Computing Spearman correlation of the DE z-scores")
spearman_per_group = []
for z_s, z_p in zip(z_sol, z_pred):
  # zero-variance z-scores contribute 0 rather than NaN
  if np.ptp(z_s) == 0 or np.ptp(z_p) == 0:
    spearman_per_group.append(0.0)
  else:
    spearman_per_group.append(stats.spearmanr(z_s, z_p)[0])
spearman_z = float(np.mean(spearman_per_group))

logging.info("Create output object")
out = ad.AnnData(
  uns = {
    "dataset_id" : ad_pred.uns["dataset_id"],
    "method_id" : ad_pred.uns["method_id"],
    "metric_ids" : ["deg_f1", "spearman_z"],
    "metric_values" : [deg_f1, spearman_z],
  }
)

logging.info("Write output to h5ad file")
out.write_h5ad(par["output"], compression=9)
