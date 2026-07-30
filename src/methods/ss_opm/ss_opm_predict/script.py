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
    'cell_type_col': None,
    'day_pattern': r'd(\d+)',
}
meta = {
    'name': 'ss_opm_predict',
    'resources_dir': 'src/methods/ss_opm',
}
## VIASH END

sys.path.append(meta['resources_dir'])
from ss_opm_common import build_metadata, to_sparse_csr

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
test_metadata = build_metadata(
    input_test_mod1,
    task_type,
    cell_type_col=par['cell_type_col'],
    group_by_batch=False,
    day_pattern=par['day_pattern'],
)

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
# Prediction must be a sparse matrix to be compatible with all metrics.
if not scipy.sparse.issparse(y_pred):
    y_pred = scipy.sparse.csr_matrix(y_pred)

output = ad.AnnData(
    layers={"normalized": y_pred},
    obs=input_test_mod1.obs,
    var=mod2_var,
    uns={
        "dataset_id": dataset_id,
        "method_id": "ss_opm",
    },
)
output.write_h5ad(par['output'], compression="gzip")
print('Done!', flush=True)
