import sys
import os
import gc
import pickle
import tempfile
import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad

from ss_opm.pre_post_processing.pre_post_processing import PrePostProcessing
from ss_opm.model.encoder_decoder.encoder_decoder import EncoderDecoder
from ss_opm.utility.set_seed import set_seed

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}', flush=True)

## VIASH START
par = {
    'input_train_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod1.h5ad',
    'input_train_mod2': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod2.h5ad',
    'input_test_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad',
    'output': 'output/models/ss_opm',
    'cell_type_col': None,
    'day_pattern': r'd(\d+)',
    'n_epochs': 40,
    'burnin_length_epoch': 10,
}
meta = {
    'name': 'ss_opm_train',
    'resources_dir': 'src/methods/ss_opm',
}
## VIASH END

sys.path.append(meta['resources_dir'])
from ss_opm_common import build_metadata, to_sparse_csr

# Monkey-patch the source utility modules so ALL callers (pre_post_processing,
# IterativeSVDImputator, etc.) use safe versions that handle all-zero/all-NaN rows.
import ss_opm.utility.nonzero_median_normalize as _mnm_module
import ss_opm.utility.row_normalize as _rn_module
import ss_opm.pre_post_processing.pre_post_processing as _pp_module

def _safe_median_normalize(values, ignore_zero=True, log=False):
    """Median-normalize rows, substituting 1 (identity) when the median is 0 or NaN."""
    arr = np.asarray(values.toarray() if hasattr(values, 'toarray') else values, dtype=float).copy()
    for_median = arr.copy()
    if ignore_zero:
        for_median[for_median == 0] = np.nan
    med = np.nanquantile(for_median, q=0.5, axis=1)
    # Use 1 as fallback so rows with zero/undefined median are left unchanged
    med = np.where((med == 0) | ~np.isfinite(med), 1.0, med)
    if log:
        return arr - med[:, None]
    else:
        return arr / med[:, None]

def _safe_row_normalize(v):
    """Row-standardize; rows with std=0 are mean-subtracted only (result is zeros)."""
    mu = np.mean(v, axis=1)
    sigma = np.std(v, axis=1)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (v - mu[:, None]) / sigma[:, None]

# Patch the source modules so every 'from X import Y' binding stays in sync
_mnm_module.median_normalize = _safe_median_normalize
_rn_module.row_normalize = _safe_row_normalize
# Also patch the names already bound inside pre_post_processing's namespace
_pp_module.median_normalize = _safe_median_normalize
_pp_module.row_normalize = _safe_row_normalize

# The SVD decomposer components are stored as float64 tensors inside
# MultiEncoderDecoderModule, but the neural-network outputs are float32.
# Patch _train_step_forward to convert the whole sub-model to float32
# immediately before any forward pass, so all tensors share the same dtype.
import ss_opm.model.encoder_decoder.encoder_decoder as _ed_module

_orig_train_step_fwd = _ed_module.EncoderDecoder._train_step_forward

def _patched_train_step_fwd(self, batch, training_length_ratio):
    if hasattr(self, 'model') and self.model is not None:
        self.model.float()
    return _orig_train_step_fwd(self, batch, training_length_ratio)

_ed_module.EncoderDecoder._train_step_forward = _patched_train_step_fwd

SEED = 42
set_seed(SEED)


# ---- Load data ----
print('Loading data...', flush=True)
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])

mod1 = input_train_mod1.uns['modality']
mod2 = input_train_mod2.uns['modality']
dataset_id = input_train_mod1.uns['dataset_id']
print(f'Modalities: {mod1} -> {mod2}', flush=True)

# Determine task type: 'cite' when ADT is involved, 'multi' for ATAC/GEX
task_type = 'cite' if 'ADT' in (mod1, mod2) else 'multi'
print(f'Task type: {task_type}', flush=True)

train_inputs = to_sparse_csr(input_train_mod1.layers['normalized'])
train_targets = to_sparse_csr(input_train_mod2.layers['normalized'])
n_vars_mod1 = train_inputs.shape[1]
n_vars_mod2 = train_targets.shape[1]

train_metadata = build_metadata(
    input_train_mod1,
    task_type,
    cell_type_col=par['cell_type_col'],
    day_pattern=par['day_pattern'],
)

# Store mod2 var for the predict step
mod2_var = input_train_mod2.var.copy()

del input_train_mod1, input_train_mod2
gc.collect()

# ---- Load test inputs for SVD fitting (optional but improves preprocessing) ----
test_inputs = None
test_metadata = None
if par.get('input_test_mod1'):
    print('Loading test data for SVD fitting...', flush=True)
    input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
    test_inputs = to_sparse_csr(input_test_mod1.layers['normalized'])
    test_metadata = build_metadata(
        input_test_mod1,
        task_type,
        cell_type_col=par['cell_type_col'],
        day_pattern=par['day_pattern'],
    )
    del input_test_mod1
    gc.collect()

# ---- Create data_dir with CITE-specific mask files ----
# The original PrePostProcessing loads pre-computed feature-target correlation
# masks from data_dir. We replace them with all-True masks so all input features
# are retained as supplementary raw features alongside the SVD components.
data_dir = tempfile.mkdtemp()
if task_type == 'cite':
    mask_pair = np.ones((n_vars_mod1, n_vars_mod2), dtype=bool)
    np.savez(os.path.join(data_dir, 'cite_inputs_targets_pair3g.npz'), mask=mask_pair)
    mask2 = np.zeros((n_vars_mod1,), dtype=bool)
    np.savez(os.path.join(data_dir, 'cite_inputs_mask2.npz'), mask=mask2)

# ---- Get parameters ----
pre_post_process_params = PrePostProcessing.get_params(
    task_type=task_type,
    data_dir=data_dir,
    device=device,
    seed=SEED,
)
model_params = EncoderDecoder.get_params(
    task_type=task_type,
    device=device,
)
model_params['epoch'] = par['n_epochs']
model_params['burnin_length_epoch'] = par['burnin_length_epoch']

# ---- Fit preprocessing ----
print('Fitting preprocessing...', flush=True)
pre_post_process = PrePostProcessing(pre_post_process_params)

# Use test inputs alongside train inputs for fitting SVD (improves coverage)
_test_inputs_for_svd = test_inputs if test_inputs is not None else train_inputs
_test_metadata_for_svd = test_metadata if test_metadata is not None else train_metadata

pre_post_process.fit_preprocess(
    inputs_values=train_inputs,
    targets_values=train_targets,
    metadata=train_metadata,
    test_inputs_values=_test_inputs_for_svd,
    test_metadata=_test_metadata_for_svd,
)

# ---- Preprocess training data ----
print('Preprocessing training data...', flush=True)
preprocessed_inputs, preprocessed_targets = pre_post_process.preprocess(
    inputs_values=train_inputs,
    targets_values=train_targets,
    metadata=train_metadata,
)

# ---- Train model ----
# Cast preprocessed arrays to float32 to match what PyTorch expects.
if isinstance(preprocessed_inputs, np.ndarray):
    preprocessed_inputs = preprocessed_inputs.astype(np.float32)
if isinstance(preprocessed_targets, np.ndarray):
    preprocessed_targets = preprocessed_targets.astype(np.float32)

print('Training model...', flush=True)
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

# ---- Save model and preprocessing artifacts ----
print('Saving model...', flush=True)
os.makedirs(par['output'], exist_ok=True)

model_dir = os.path.join(par['output'], 'model')
os.makedirs(model_dir, exist_ok=True)
model.save(model_dir)

with open(os.path.join(par['output'], 'pre_post_process.pickle'), 'wb') as f:
    pickle.dump(pre_post_process, f)

mod2_var.to_parquet(os.path.join(par['output'], 'mod2_var.parquet'))

with open(os.path.join(par['output'], 'task_info.pickle'), 'wb') as f:
    pickle.dump({'task_type': task_type, 'mod2': mod2, 'dataset_id': dataset_id}, f)

print('Done!', flush=True)
