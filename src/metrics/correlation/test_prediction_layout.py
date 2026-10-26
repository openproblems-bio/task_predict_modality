import subprocess

import anndata as ad
import numpy as np
import scipy.sparse
from scipy.stats import rankdata

## VIASH START
meta = {
    "executable": "target/executable/metrics/correlation/correlation",
    "resources_dir": ".",
}
## VIASH END

# A prediction must get the same scores whether its layer is stored as CSR or CSC, and its per-cell and per-gene
# Spearman correlations must match those of dense ranks. dynutils' spearman_rank_sparse() ranks along the compressed
# dimension, so a CSR prediction used to be ranked along the wrong axis.
input_dir = f"{meta['resources_dir']}/resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap"
solution = ad.read_h5ad(f"{input_dir}/test_mod2.h5ad")
prediction = ad.read_h5ad(f"{input_dir}/prediction.h5ad")


def score(layout):
    prediction.layers["normalized"] = scipy.sparse.csr_matrix(prediction.layers["normalized"]).asformat(layout)
    prediction.write_h5ad(f"prediction_{layout}.h5ad")
    subprocess.run(
        [
            meta["executable"],
            "--input_test_mod2", f"{input_dir}/test_mod2.h5ad",
            "--input_prediction", f"prediction_{layout}.h5ad",
            "--output", f"scores_{layout}.h5ad",
        ],
        check=True,
    )
    return ad.read_h5ad(f"scores_{layout}.h5ad")


def paired_spearman(truth, predicted, axis):
    """Spearman correlation of matching rows (axis=1) or columns (axis=0); constant vectors score 0."""
    truth_ranks, predicted_ranks = (rankdata(values, axis=axis) for values in (truth, predicted))
    truth_ranks -= truth_ranks.mean(axis=axis, keepdims=True)
    predicted_ranks -= predicted_ranks.mean(axis=axis, keepdims=True)
    norms = np.sqrt((truth_ranks**2).sum(axis=axis) * (predicted_ranks**2).sum(axis=axis))
    covariance = (truth_ranks * predicted_ranks).sum(axis=axis)
    return np.divide(covariance, norms, out=np.zeros_like(covariance), where=norms > 0)


scores = {layout: score(layout) for layout in ["csr", "csc"]}
truth = solution.layers["normalized"].toarray()
predicted = prediction.layers["normalized"].toarray()
expected_spearman = {"obs": paired_spearman(truth, predicted, axis=1), "var": paired_spearman(truth, predicted, axis=0)}

for layout, layout_scores in scores.items():
    print(f"{layout}:", dict(zip(layout_scores.uns["metric_ids"], np.round(layout_scores.uns["metric_values"], 4))))
    np.testing.assert_allclose(layout_scores.uns["metric_values"], scores["csc"].uns["metric_values"], rtol=0, atol=1e-10)
    for annotation, expected in expected_spearman.items():
        np.testing.assert_allclose(getattr(layout_scores, annotation)["spearman"], expected, rtol=0, atol=1e-8)
