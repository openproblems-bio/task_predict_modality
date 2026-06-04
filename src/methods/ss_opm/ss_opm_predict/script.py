import sys
import os
import gc
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad
from ss_opm.model.encoder_decoder.encoder_decoder import EncoderDecoder

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}', flush=True)

## VIASH START
par = {
    'input_test_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad',
    'input_model': 'output/models/ss_opm',
    'output': 'output/prediction.h5ad',
}
meta = {
    'name': 'ss_opm_predict',
    'resources_dir': 'src/methods/ss_opm/ss_opm_predict',
}
## VIASH END

def build_metadata(adata, task_type):
    """Build a metadata DataFrame compatible with ss_opm from an H5AD AnnData.

    Mirrors the function in the train script; only used for the input preprocessing
    path (targets are None at predict time, so group IDs are not critical).
    """
    obs = pd.DataFrame(index=adata.obs_names)

    obs['batch'] = adata.obs['batch'].values
    obs['day'] = adata.obs['batch'].str.extract(r'd(\d+)', expand=False).astype(float).fillna(0).values

    X = adata.layers['normalized']
    if scipy.sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X, dtype=float)

    obs['nonzero_ratio'] = (X_dense != 0).mean(axis=1)
    obs['nonzero_q25'] = np.percentile(X_dense, 25, axis=1)
    obs['nonzero_q50'] = np.percentile(X_dense, 50, axis=1)
    obs['nonzero_q75'] = np.percentile(X_dense, 75, axis=1)
    obs['mean'] = X_dense.mean(axis=1)
    obs['std'] = X_dense.std(axis=1)

    # Group: 0 for all test cells (group is not used during input-only preprocessing)
    obs['group'] = 0
    obs['cell_type'] = 'hidden'
    obs['donor'] = 0
    obs['technology'] = 'unknown'

    if task_type == 'cite':
        for ct in ['HSC', 'EryP', 'NeuP', 'MasP', 'MkP', 'BP', 'MoP']:
            obs[f'cell_ratio_{ct}'] = 1.0 / 7
        obs['cell_count'] = float(adata.n_obs)
        for i in range(8):
            obs[f'batch_sv{i}'] = 0.0

    return obs


def to_sparse_csr(X):
    if scipy.sparse.issparse(X):
        return X.tocsr()
    return scipy.sparse.csr_matrix(X)


# ---- Load task info ----
with open(os.path.join(par['input_model'], 'task_info.pickle'), 'rb') as f:
    task_info = pickle.load(f)
task_type = task_info['task_type']
mod2 = task_info['mod2']
dataset_id = task_info['dataset_id']
print(f'Task type: {task_type}, mod2: {mod2}', flush=True)

# ---- Load test data ----
print('Loading test data...', flush=True)
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
test_inputs = to_sparse_csr(input_test_mod1.layers['normalized'])
test_metadata = build_metadata(input_test_mod1, task_type)

# ---- Load model and preprocessing artifacts ----
print('Loading model...', flush=True)
with open(os.path.join(par['input_model'], 'pre_post_process.pickle'), 'rb') as f:
    pre_post_process = pickle.load(f)

model = EncoderDecoder(params=None)
# PyTorch >=2.6 defaults weights_only=True, which blocks custom classes.
# Patch torch.load to use weights_only=False for trusted local model files.
import torch as _torch
_orig_torch_load = _torch.load
_torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, 'weights_only': False})
model.load(os.path.join(par['input_model'], 'model'))
_torch.load = _orig_torch_load
model.params['device'] = device

mod2_var = pd.read_parquet(os.path.join(par['input_model'], 'mod2_var.parquet'))

# ---- Preprocess test inputs ----
print('Preprocessing test data...', flush=True)
preprocessed_test_inputs, _ = pre_post_process.preprocess(
    inputs_values=test_inputs,
    targets_values=None,
    metadata=test_metadata,
)

# ---- Predict ----
print('Predicting...', flush=True)
y_pred = model.predict(
    x=test_inputs,
    preprocessed_x=preprocessed_test_inputs,
    metadata=test_metadata,
)
gc.collect()

# ---- Write output ----
print('Writing output...', flush=True)
output = ad.AnnData(
    layers={"normalized": y_pred},
    obs=input_test_mod1.obs,
    var=mod2_var,
    uns={
        "dataset_id": dataset_id,
        "method_id": meta["name"],
    },
)
output.write_h5ad(par['output'], compression="gzip")
print('Done!', flush=True)
