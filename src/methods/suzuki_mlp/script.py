import sys
import logging
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import gc
import warnings
warnings.filterwarnings('ignore')

## VIASH START
par = {
    'input_train_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod1.h5ad',
    'input_train_mod2': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/train_mod2.h5ad',
    'input_test_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/swap/test_mod1.h5ad',
    'output': 'output.h5ad',
    'task_type': 'auto',
    'inputs_n_components': 128,
    'targets_n_components': 128,
    'encoder_h_dim': 512,
    'decoder_h_dim': 512,
    'n_encoder_block': 3,
    'n_decoder_block': 3,
    'dropout_p': 0.1,
    'activation': 'relu',
    'norm': 'layer_norm',
    'use_skip_connections': True,
    'learning_rate': 0.0001,
    'weight_decay': 0.000001,
    'epochs': 40,
    'batch_size': 64,
    'use_residual_connections': True,
}
meta = {
    'name': 'suzuki_mlp'
}
## VIASH END

# Import utils functions
import sys
import os
sys.path.append(meta["resources_dir"])

from utils import (
    determine_task_type, preprocess_data, train_model,
    MLPBModule, HierarchicalMLPBModule, SuzukiEncoderDecoderModule
)

def main():
    # Enable logging
    logging.basicConfig(level=logging.INFO)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    # Read input files
    print("Reading input files", flush=True)
    adata_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
    adata_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
    adata_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
    
    # Determine task type
    if par['task_type'] == 'auto':
        task_type = determine_task_type(adata_train_mod1, adata_train_mod2)
        print(f"Auto-detected task type: {task_type}", flush=True)
    else:
        task_type = par['task_type']
    
    print(f"Task type: {task_type}", flush=True)
    print(f"Modality 1: {adata_train_mod1.uns.get('modality', 'Unknown')}, n_features: {adata_train_mod1.n_vars}")
    print(f"Modality 2: {adata_train_mod2.uns.get('modality', 'Unknown')}, n_features: {adata_train_mod2.n_vars}")
    
    # Preprocess data
    print("Preprocessing data", flush=True)
    data = preprocess_data(
        adata_train_mod1=adata_train_mod1,
        adata_train_mod2=adata_train_mod2,
        adata_test_mod1=adata_test_mod1,
        task_type=task_type,
        inputs_n_components=par['inputs_n_components'],
        targets_n_components=par['targets_n_components']
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    metadata_train = data['metadata_train']
    metadata_test = data['metadata_test']
    targets_decomposer_components = data['targets_decomposer_components']
    targets_global_median = data['targets_global_median']
    y_statistic = data['y_statistic']
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}")
    
    # Build model
    print("Building model", flush=True)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    # Create encoder
    encoder = MLPBModule(
        input_dim=None,  # Will be set in the main module
        output_dim=par['encoder_h_dim'],
        n_block=par['n_encoder_block'],
        h_dim=par['encoder_h_dim'],
        skip=par['use_skip_connections'],
        dropout_p=par['dropout_p'],
        activation=par['activation'],
        norm="layer_norm"
    )
    
    # Create hierarchical decoder
    decoder = HierarchicalMLPBModule(
        input_dim=par['encoder_h_dim'],
        output_dim=None,  # Will create multiple outputs
        n_block=par['n_decoder_block'],
        h_dim=par['decoder_h_dim'],
        skip=par['use_skip_connections'],
        dropout_p=par['dropout_p'],
        activation=par['activation'],
        norm="layer_norm"
    )
    
    # Create main model
    model = SuzukiEncoderDecoderModule(
        x_dim=input_dim,
        y_dim=output_dim,
        y_statistic=y_statistic,
        encoder_h_dim=par['encoder_h_dim'],
        decoder_h_dim=par['decoder_h_dim'],
        n_decoder_block=par['n_decoder_block'],
        targets_decomposer_components=targets_decomposer_components,
        targets_global_median=targets_global_median,
        encoder=encoder,
        decoder=decoder,
        task_type=task_type,
        use_residual_connections=par['use_residual_connections']
    ).to(device)
    
    # Train model
    print("Training model", flush=True)
    trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        metadata_train=metadata_train,
        device=device,
        lr=par['learning_rate'],
        weight_decay=par['weight_decay'],
        epochs=par['epochs'],
        batch_size=par['batch_size'],
        task_type=task_type
    )
    
    # Predict on test data
    print("Predicting on test data", flush=True)
    trained_model.eval()
    predictions = []
    
    with torch.no_grad():
        # Handle metadata safely for test data
        if 'gender' in metadata_test.columns:
            gender_values = metadata_test['gender'].values
            if gender_values.dtype == object:
                gender_values = pd.to_numeric(gender_values, errors='coerce').fillna(0).astype(int)
            gender_test = torch.LongTensor(gender_values)
        else:
            gender_test = torch.LongTensor(np.zeros(len(X_test), dtype=int))
        
        info_test = torch.FloatTensor(np.zeros((len(X_test), 1)))
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            gender_test,
            info_test
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=par['batch_size'], shuffle=False)
        
        for batch_x, batch_gender, batch_info in test_loader:
            batch_x = batch_x.to(device)
            batch_gender = batch_gender.to(device)
            batch_info = batch_info.to(device)
            
            pred = trained_model.predict(batch_x, batch_gender, batch_info)
            predictions.append(pred.cpu().numpy())
    
    y_pred = np.vstack(predictions)
    
    # Create output AnnData object
    print("Creating output", flush=True)
    adata_pred = ad.AnnData(
        obs=adata_test_mod1.obs.copy(),
        var=adata_train_mod2.var.copy(),
        layers={
            'normalized': y_pred
        },
        uns={
            'dataset_id': adata_train_mod1.uns.get('dataset_id', 'unknown'),
            'method_id': meta['name']
        }
    )
    
    # Write output
    print("Writing output to file", flush=True)
    adata_pred.write_h5ad(par['output'], compression='gzip')
    
    print("Done!", flush=True)

if __name__ == '__main__':
    main()
