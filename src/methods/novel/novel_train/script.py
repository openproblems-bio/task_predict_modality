import sys
import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split

import anndata as ad
import pickle

#check gpu available
if (torch.cuda.is_available()):
    device = 'cuda:0' #switch to current device
    print('current device: gpu', flush=True)
else:
    device = 'cpu'
    print('current device: cpu', flush=True)


## VIASH START
par = {
  'input_train_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/normal/train_mod1.h5ad',
  'input_train_mod2': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/normal/train_mod2.h5ad',
  'output': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/normal/models/novel'
}
meta = {
  'resources_dir': 'src/methods/novel',
}
## VIASH END

sys.path.append(meta['resources_dir'])
from helper_functions import train_and_valid, lsiTransformer, ModalityMatchingDataset
from helper_functions import ModelRegressionAtac2Gex, ModelRegressionAdt2Gex, ModelRegressionGex2Adt, ModelRegressionGex2Atac

print('Load data', flush=True)
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])

adata = input_train_mod2.copy()

mod1 = input_train_mod1.uns['modality']
mod2 = input_train_mod2.uns['modality']

input_train_mod1.X = input_train_mod1.layers['normalized']
input_train_mod2.X = input_train_mod2.layers['normalized']

input_train_mod2_df = input_train_mod2.to_df()

del input_train_mod2

print('Start train', flush=True)
# Check for zero divide
zero_row = input_train_mod1.X.sum(axis=0) == 0

rem_var = None
if True in zero_row:
  rem_var = input_train_mod1[:, zero_row].var_names
  input_train_mod1 = input_train_mod1[:, ~zero_row]


# select number of variables for LSI
n_comp = input_train_mod1.n_vars -1 if input_train_mod1.n_vars < 256 else 256

if mod1 != 'ADT':  
  lsi_transformer_gex = lsiTransformer(n_components=n_comp)
  input_train_mod1_df = lsi_transformer_gex.fit_transform(input_train_mod1)
else:
  input_train_mod1_df = input_train_mod1.to_df()

# reproduce train/test split from phase 1
batch = input_train_mod1.obs["batch"]
test_batches = {'s1d2', 's3d7'}
# if none of phase1_batch is in batch, sample 25% of batch categories rounded up
if len(test_batches.intersection(set(batch))) == 0:
  all_batches = batch.cat.categories.tolist()
  test_batches = set(np.random.choice(all_batches, math.ceil(len(all_batches) * 0.25), replace=False))
train_ix = [ k for k,v in enumerate(batch) if v not in test_batches ]
test_ix = [ k for k,v in enumerate(batch) if v in test_batches ]

train_mod1 = input_train_mod1_df.iloc[train_ix, :]
train_mod2 = input_train_mod2_df.iloc[train_ix, :]
test_mod1 = input_train_mod1_df.iloc[test_ix, :]
test_mod2 = input_train_mod2_df.iloc[test_ix, :]

n_vars_train_mod1 = train_mod1.shape[1]
n_vars_train_mod2 = train_mod2.shape[1]
n_vars_test_mod1 = test_mod1.shape[1]
n_vars_test_mod2 = test_mod2.shape[1]

n_vars_mod1 = input_train_mod1_df.shape[1]
n_vars_mod2 = input_train_mod2_df.shape[1]

if mod1 == 'ATAC' and mod2 == 'GEX':
  dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
  dataloader_train = DataLoader(dataset_train, 256, shuffle = True, num_workers = 8)

  dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
  dataloader_test = DataLoader(dataset_test, 64, shuffle = False, num_workers = 8)

  model = ModelRegressionAtac2Gex(n_vars_mod1,n_vars_mod2).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.00008386597445284492,weight_decay=0.000684887347727808)

elif mod1 == 'ADT' and mod2 == 'GEX':
  dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
  dataloader_train = DataLoader(dataset_train, 64, shuffle = True, num_workers = 4)

  dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
  dataloader_test = DataLoader(dataset_test, 32, shuffle = False, num_workers = 4)

  model = ModelRegressionAdt2Gex(n_vars_mod1,n_vars_mod2).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00041, weight_decay=0.0000139)


elif mod1 == 'GEX' and mod2 == 'ADT':
  dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
  dataloader_train = DataLoader(dataset_train, 32, shuffle = True, num_workers = 8)

  dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
  dataloader_test = DataLoader(dataset_test, 64, shuffle = False, num_workers = 8)

  model = ModelRegressionGex2Adt(n_vars_mod1,n_vars_mod2).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.000034609210829678734, weight_decay=0.0009965881574697426)


elif mod1 == 'GEX' and mod2 == 'ATAC':
  dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
  dataloader_train = DataLoader(dataset_train, 64, shuffle = True, num_workers = 8)

  dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
  dataloader_test = DataLoader(dataset_test, 64, shuffle = False, num_workers = 8)

  model = ModelRegressionGex2Atac(n_vars_mod1,n_vars_mod2).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001806762345275399, weight_decay=0.0004084171379280058)

loss_fn = torch.nn.MSELoss()

# create dir for par['output']
os.makedirs(par['output'], exist_ok=True)

# determine filenames
output_model = f"{par['output']}/tensor.pt"
output_h5ad = f"{par['output']}/train_mod2.h5ad"
output_transform = f"{par['output']}/transform.pkl"

# train model
train_and_valid(model, optimizer, loss_fn, dataloader_train, dataloader_test, output_model, device)

# Add model dim for use in predict part
adata.uns["model_dim"] = {"mod1": n_vars_mod1, "mod2": n_vars_mod2}
if rem_var is not None:
  adata.uns["removed_vars"] = [rem_var[0]]
adata.write_h5ad(output_h5ad, compression="gzip")

if mod1 != 'ADT':
    with open(output_transform, 'wb') as f:
        pickle.dump(lsi_transformer_gex, f)
