import anndata as ad
import logging
import numpy as np
from scipy.sparse import csr_matrix

## VIASH START
par = {
  "input_test_mod2" : "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/test_mod2.h5ad",
  "input_prediction" : "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/normal/prediction.h5ad",
  "output" : "output/scores.h5ad"
}
## VIASH END

logging.info("Reading solution file")
ad_sol = ad.read_h5ad(par["input_test_mod2"])

logging.info("Reading prediction file")
ad_pred = ad.read_h5ad(par["input_prediction"])

logging.info("Check prediction format")
if ad_sol.uns["dataset_id"] != ad_pred.uns["dataset_id"]:
  raise ValueError("Prediction and solution have differing dataset_ids")

if ad_sol.shape != ad_pred.shape:
  raise ValueError("Dataset and prediction anndata objects should have the same shape / dimensions.")

logging.info("Computing MSE metrics")

# coerce to sparse -- sparse minus dense yields a np.matrix, which has no .power()
sol = csr_matrix(ad_sol.layers["normalized"])
pred = csr_matrix(ad_pred.layers["normalized"])

# score non-finite predictions as zero, rather than returning a NaN metric
non_finite = ~np.isfinite(pred.data)
if non_finite.any():
  logging.info("Prediction contains %d non-finite values, scoring them as 0", non_finite.sum())
  pred.data[non_finite] = 0

tmp = sol - pred
rmse = np.sqrt(tmp.power(2).mean())
mae = np.abs(tmp).mean()

logging.info("Create output object")
out = ad.AnnData(
  uns = {
    "dataset_id" : ad_pred.uns["dataset_id"],
    "method_id" : ad_pred.uns["method_id"],
    "metric_ids" : ["rmse", "mae"],
    "metric_values" : [rmse, mae],
  }
)

logging.info("Write output to h5ad file")
out.write_h5ad(par["output"], compression=9)
