import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import anndata as ad
from typing import Dict, Any, Optional, Tuple
import gc

def determine_task_type(adata_mod1: ad.AnnData, adata_mod2: ad.AnnData) -> str:
    """Automatically determine the task type based on modalities."""
    mod1 = adata_mod1.uns.get('modality', 'GEX')
    mod2 = adata_mod2.uns.get('modality', 'Unknown')
    
    if mod1 == 'GEX' and mod2 == 'ADT':
        return 'cite'
    elif (mod1 == 'GEX' and mod2 == 'ATAC') or (mod1 == 'ATAC' and mod2 == 'GEX'):
        return 'multi'
    else:
        # Default fallback based on number of features
        if adata_mod2.n_vars < 500:  # ADT typically has fewer features
            return 'cite'
        else:
            return 'multi'

def median_normalize(data: np.ndarray, ignore_zero: bool = True) -> np.ndarray:
    """Median normalization as used in Suzuki's solution."""
    if sparse.issparse(data):
        data = data.toarray()
    
    if ignore_zero:
        # Only consider non-zero values for median calculation
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            row = data[i]
            nonzero_vals = row[row > 0]
            if len(nonzero_vals) > 0:
                median_val = np.median(nonzero_vals)
                if median_val > 0:
                    result[i] = row / median_val
                else:
                    result[i] = row
            else:
                result[i] = row
        return result
    else:
        medians = np.median(data, axis=1, keepdims=True)
        medians[medians == 0] = 1  # Avoid division by zero
        return data / medians

def row_quantile_normalize(data: np.ndarray, q: float = 0.75) -> np.ndarray:
    """Row-wise quantile normalization."""
    if sparse.issparse(data):
        data = data.toarray()
    
    quantiles = np.quantile(data, q, axis=1, keepdims=True)
    quantiles[quantiles == 0] = 1
    return data / quantiles

def row_normalize(data: np.ndarray) -> np.ndarray:
    """Z-score normalize each row (cell)."""
    if isinstance(data, torch.Tensor):
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        std[std == 0] = 1  # Avoid division by zero
        return (data - mean) / std
    else:
        if sparse.issparse(data):
            data = data.toarray()
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1
        return (data - mean) / std

def preprocess_data(
    adata_train_mod1: ad.AnnData,
    adata_train_mod2: ad.AnnData,
    adata_test_mod1: ad.AnnData,
    task_type: str,
    inputs_n_components: int = 128,
    targets_n_components: int = 128
) -> Dict[str, Any]:
    """Preprocess data following Suzuki's approach."""
    
    # Extract data
    X_train = adata_train_mod1.layers['normalized'].toarray() if sparse.issparse(adata_train_mod1.layers['normalized']) else adata_train_mod1.layers['normalized']
    y_train = adata_train_mod2.layers['normalized'].toarray() if sparse.issparse(adata_train_mod2.layers['normalized']) else adata_train_mod2.layers['normalized']
    X_test = adata_test_mod1.layers['normalized'].toarray() if sparse.issparse(adata_test_mod1.layers['normalized']) else adata_test_mod1.layers['normalized']
    
    # Task-specific preprocessing for inputs
    if task_type == 'multi':
        # For multiome data, use quantile normalization
        X_combined = np.vstack([X_train, X_test])
        X_combined = row_quantile_normalize(X_combined)
        X_train = X_combined[:len(X_train)]
        X_test = X_combined[len(X_train):]
        
        # Targets preprocessing for multi
        y_train = median_normalize(y_train, ignore_zero=False)
        y_train = row_normalize(y_train)
        
    elif task_type == 'cite':
        # For CITE-seq data, use log1p + median normalization
        X_combined = np.vstack([X_train, X_test])
        X_combined = np.log1p(median_normalize(np.expm1(X_combined)))
        X_train = X_combined[:len(X_train)]
        X_test = X_combined[len(X_train):]
        
        # Targets preprocessing for cite
        y_train = np.log1p(median_normalize(np.expm1(y_train)))
        y_train = row_normalize(y_train)
    
    # Apply SVD to inputs
    print(f"Applying SVD to inputs (n_components={inputs_n_components})")
    inputs_decomposer = TruncatedSVD(n_components=inputs_n_components, random_state=42)
    X_combined = np.vstack([X_train, X_test])
    X_combined_transformed = inputs_decomposer.fit_transform(X_combined)
    X_train = X_combined_transformed[:len(X_train)]
    X_test = X_combined_transformed[len(X_train):]
    
    # Apply SVD to targets
    print(f"Applying SVD to targets (n_components={targets_n_components})")
    targets_decomposer = TruncatedSVD(n_components=targets_n_components, random_state=42)
    y_train_transformed = targets_decomposer.fit_transform(y_train)
    
    # Calculate batch-wise medians for targets (simplified version)
    targets_global_median = np.median(y_train, axis=0)
    y_train_centered = y_train - targets_global_median[None, :]
    
    # Calculate target statistics
    y_statistic = {
        'y_loc': np.mean(y_train_transformed, axis=0),
        'y_scale': np.std(y_train_transformed, axis=0),
        'targets_global_median': targets_global_median
    }
    
    # Prepare metadata (simplified)
    metadata_train = adata_train_mod1.obs.copy()
    metadata_train['gender'] = 0  # Simplified - could extract from actual metadata
    metadata_test = adata_test_mod1.obs.copy()  
    metadata_test['gender'] = 0
    
    return {
        'X_train': X_train,
        'y_train': y_train_transformed,
        'X_test': X_test,
        'metadata_train': metadata_train,
        'metadata_test': metadata_test,
        'targets_decomposer_components': targets_decomposer.components_,
        'targets_global_median': targets_global_median,
        'y_statistic': y_statistic
    }

# Neural Network Components (following Suzuki's architecture)

class LinearBlock(nn.Module):
    """Linear block with activation, normalization, and dropout."""
    
    def __init__(self, h_dim: int = 128, skip: bool = False, dropout_p: float = 0.1, 
                 activation: str = "relu", norm: str = "layer_norm"):
        super(LinearBlock, self).__init__()
        self.skip = skip
        self.fc = nn.Linear(h_dim, h_dim, bias=False)
        
        # Normalization
        if norm == "batch_norm":
            self.norm = nn.BatchNorm1d(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        elif norm == "layer_norm":
            self.norm = nn.LayerNorm(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        else:
            self.norm = None
        
        # Dropout
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None
        
        # Activation
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        h = x
        h_prev = x
        h = self.act(h)
        if self.norm is not None:
            h = self.norm(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.fc(h)
        if self.skip:
            h = h + h_prev
        return h

class MLPBModule(nn.Module):
    """Multi-layer perceptron with blocks."""
    
    def __init__(self, input_dim: Optional[int], output_dim: Optional[int], n_block: int, 
                 h_dim: int = 128, skip: bool = False, dropout_p: float = 0.1, 
                 activation: str = "relu", norm: str = "layer_norm"):
        super(MLPBModule, self).__init__()
        
        self.in_fc = None
        if input_dim is not None:
            self.in_fc = nn.Linear(input_dim, h_dim)
        
        layers = []
        for _ in range(n_block):
            layers.append(LinearBlock(h_dim=h_dim, skip=skip, dropout_p=dropout_p, 
                                    activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        
        self.out_fc = None
        if output_dim is not None:
            self.out_fc = nn.Linear(h_dim, output_dim)
    
    def forward(self, x):
        h = x
        if self.in_fc is not None:
            h = self.in_fc(h)
        for layer in self.layers:
            h = layer(h)
        if self.out_fc is not None:
            y = self.out_fc(h)
        else:
            y = h
        return y, h

class HierarchicalMLPBModule(nn.Module):
    """Hierarchical MLP that outputs from multiple layers."""
    
    def __init__(self, input_dim: Optional[int], output_dim: Optional[int], n_block: int,
                 h_dim: int = 128, skip: bool = False, dropout_p: float = 0.1,
                 activation: str = "relu", norm: str = "layer_norm"):
        super(HierarchicalMLPBModule, self).__init__()
        
        self.in_fc = None
        if input_dim is not None:
            self.in_fc = nn.Linear(input_dim, h_dim)
        
        layers = []
        for _ in range(n_block):
            layers.append(LinearBlock(h_dim=h_dim, skip=skip, dropout_p=dropout_p,
                                    activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        
        self.out_fc = None
        if output_dim is not None:
            self.out_fc = nn.Linear(h_dim, output_dim)
    
    def forward(self, x):
        h = x
        if self.in_fc is not None:
            h = self.in_fc(h)
        hs = [h]
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        if self.out_fc is not None:
            y = self.out_fc(h)
        else:
            y = h
        return y, hs

def correlation_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Correlation loss function."""
    # Center the predictions and targets
    y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
    y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
    
    # Calculate correlation
    numerator = (y_pred_centered * y_true_centered).sum(dim=0)
    denominator = torch.sqrt((y_pred_centered ** 2).sum(dim=0) * (y_true_centered ** 2).sum(dim=0))
    
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-8)
    correlation = numerator / denominator
    
    # Return negative correlation as loss (we want to maximize correlation)
    return -correlation.mean()

class SuzukiEncoderDecoderModule(nn.Module):
    """Main encoder-decoder module following Suzuki's architecture."""
    
    def __init__(self, x_dim: int, y_dim: int, y_statistic: Dict,
                 encoder_h_dim: int, decoder_h_dim: int, n_decoder_block: int,
                 targets_decomposer_components: np.ndarray, targets_global_median: np.ndarray,
                 encoder: nn.Module, decoder: nn.Module, task_type: str = 'cite',
                 use_residual_connections: bool = True):
        super(SuzukiEncoderDecoderModule, self).__init__()
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.task_type = task_type
        self.use_residual_connections = use_residual_connections
        self.info_dim = 1  # Simplified metadata dimension
        
        self.encoder = encoder
        self.decoder = decoder
        
        # Statistics and components
        self.y_loc = nn.Parameter(torch.FloatTensor(y_statistic['y_loc']), requires_grad=False)
        self.y_scale = nn.Parameter(torch.FloatTensor(y_statistic['y_scale']), requires_grad=False)
        self.targets_decomposer_components = nn.Parameter(torch.FloatTensor(targets_decomposer_components), requires_grad=False)
        self.targets_global_median = nn.Parameter(torch.FloatTensor(targets_global_median), requires_grad=False)
        
        # Embeddings
        self.gender_embedding = nn.Parameter(torch.randn(2, encoder_h_dim))
        self.encoder_in_fc = nn.Linear(x_dim + self.info_dim, encoder_h_dim)
        
        # Output layers for each decoder level
        decoder_out_fcs = []
        for _ in range(n_decoder_block + 1):
            decoder_out_fcs.append(nn.Linear(decoder_h_dim, y_statistic['y_loc'].shape[0]))
        self.decoder_out_fcs = nn.ModuleList(decoder_out_fcs)
        
        # Residual output layers for multi task
        if task_type == 'multi' and use_residual_connections:
            decoder_out_res_fcs = []
            for _ in range(n_decoder_block + 1):
                decoder_out_res_fcs.append(nn.Linear(decoder_h_dim, targets_decomposer_components.shape[1]))
            self.decoder_out_res_fcs = nn.ModuleList(decoder_out_res_fcs)
        
        # Loss functions
        self.correlation_loss_func = correlation_loss
        self.mse_loss_func = nn.MSELoss()
        self.mae_loss_func = nn.L1Loss()
    
    def _encode(self, x: torch.Tensor, gender_id: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        h = torch.hstack((x, info.reshape((x.shape[0], self.info_dim))))
        h = self.encoder_in_fc(h)
        h = h + self.gender_embedding[gender_id]
        z, _ = self.encoder(h)
        return z
    
    def _decode(self, z: torch.Tensor) -> Tuple[list, Optional[list]]:
        """Decode latent representation to outputs."""
        h = z
        _, hs = self.decoder(h)
        
        ys = []
        for i, h_layer in enumerate(hs):
            y_base = self.decoder_out_fcs[i](h_layer)
            y = y_base * self.y_scale[None, :] + self.y_loc[None, :]
            ys.append(y)
        
        # Residual outputs for multi task
        y_reses = None
        if self.task_type == 'multi' and self.use_residual_connections:
            y_reses = []
            for i, h_layer in enumerate(hs):
                y_res = self.decoder_out_res_fcs[i](h_layer)
                y_reses.append(y_res)
        
        return ys, y_reses
    
    def forward(self, x: torch.Tensor, gender_id: torch.Tensor, info: torch.Tensor):
        """Forward pass."""
        z = self._encode(x, gender_id, info)
        ys, y_reses = self._decode(z)
        if self.task_type == 'multi' and self.use_residual_connections:
            return ys, y_reses
        else:
            return ys
    
    def predict(self, x: torch.Tensor, gender_id: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        if self.task_type == 'multi' and self.use_residual_connections:
            y_preds, y_res_preds = self(x, gender_id, info)
            
            postprocessed_y_pred = None
            for i in range(len(y_preds)):
                new_postprocessed_y_pred = row_normalize(
                    torch.matmul(y_preds[i], self.targets_decomposer_components) + self.targets_global_median[None, :]
                )
                new_postprocessed_y_pred += y_res_preds[i]
                new_postprocessed_y_pred = row_normalize(new_postprocessed_y_pred)
                
                if postprocessed_y_pred is None:
                    postprocessed_y_pred = new_postprocessed_y_pred
                else:
                    postprocessed_y_pred += new_postprocessed_y_pred
            
            postprocessed_y_pred /= len(y_preds)
            
        else:
            y_preds = self(x, gender_id, info)
            
            postprocessed_y_pred = None
            for i in range(len(y_preds)):
                new_postprocessed_y_pred = row_normalize(
                    torch.matmul(y_preds[i], self.targets_decomposer_components) + self.targets_global_median[None, :]
                )
                
                if postprocessed_y_pred is None:
                    postprocessed_y_pred = new_postprocessed_y_pred
                else:
                    postprocessed_y_pred += new_postprocessed_y_pred
            
            postprocessed_y_pred /= len(y_preds)
        
        return postprocessed_y_pred
    
    def compute_loss(self, x: torch.Tensor, gender_id: torch.Tensor, info: torch.Tensor,
                    y: torch.Tensor, preprocessed_y: torch.Tensor, training_length_ratio: float = 1.0) -> Dict[str, torch.Tensor]:
        """Compute loss."""
        if self.task_type == 'multi' and self.use_residual_connections:
            y_preds, y_res_preds = self(x, gender_id, info)
            
            ret = {
                "loss": 0,
                "loss_corr": 0,
                "loss_mse": 0,
                "loss_res_mse": 0,
                "loss_total_corr": 0,
            }
            
            # Convert to original space for correlation computation
            postprocessed_y = torch.matmul(preprocessed_y, self.targets_decomposer_components) + self.targets_global_median[None, :]
            normalized_y = row_normalize(postprocessed_y)
            
            for i in range(len(y_preds)):
                y_pred = y_preds[i]
                y_res_pred = y_res_preds[i]
                postprocessed_y_pred = torch.matmul(y_pred, self.targets_decomposer_components) + self.targets_global_median[None, :]
                normalized_postprocessed_y_pred_detached = row_normalize(postprocessed_y_pred.detach())
                y_res = normalized_y - normalized_postprocessed_y_pred_detached
                
                y_total_pred = normalized_postprocessed_y_pred_detached + y_res_pred
                ret["loss_corr"] += self.correlation_loss_func(postprocessed_y_pred, postprocessed_y)
                ret["loss_mse"] += self.mse_loss_func(y_pred, preprocessed_y)
                ret["loss_res_mse"] += self.mse_loss_func(y_res, y_res_pred)
                ret["loss_total_corr"] += self.correlation_loss_func(y_total_pred, normalized_y)
            
            w = (1 - training_length_ratio) ** 2
            ret["loss_corr"] /= len(y_preds)
            ret["loss"] += ret["loss_corr"]
            ret["loss_mse"] /= len(y_preds)
            ret["loss"] += w * ret["loss_mse"]
            ret["loss_res_mse"] /= len(y_preds)
            ret["loss"] += w * ret["loss_res_mse"]
            ret["loss_total_corr"] /= len(y_preds)
            ret["loss"] += ret["loss_total_corr"]
            
        else:
            y_preds = self(x, gender_id, info)
            
            ret = {
                "loss": 0,
                "loss_corr": 0,
                "loss_mae": 0,
            }
            
            # Convert to original space for correlation computation
            postprocessed_y = torch.matmul(preprocessed_y, self.targets_decomposer_components) + self.targets_global_median[None, :]
            
            for i in range(len(y_preds)):
                y_pred = y_preds[i]
                postprocessed_y_pred = torch.matmul(y_pred, self.targets_decomposer_components) + self.targets_global_median[None, :]
                ret["loss_corr"] += self.correlation_loss_func(postprocessed_y_pred, postprocessed_y)
                ret["loss_mae"] += self.mae_loss_func(y_pred, preprocessed_y)
            
            w = (1 - training_length_ratio) ** 2
            ret["loss_corr"] /= len(y_preds)
            ret["loss"] += ret["loss_corr"]
            ret["loss_mae"] /= len(y_preds)
            ret["loss"] += w * ret["loss_mae"]
        
        return ret

def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                metadata_train: pd.DataFrame, device: torch.device,
                lr: float = 1e-4, weight_decay: float = 1e-6, epochs: int = 40,
                batch_size: int = 64, task_type: str = 'cite') -> nn.Module:
    """Train the model."""
    
    # Prepare data
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    
    # Handle metadata more safely
    if 'gender' in metadata_train.columns:
        gender_values = metadata_train['gender'].values
        # Convert to numeric if it's not already
        if gender_values.dtype == object:
            gender_values = pd.to_numeric(gender_values, errors='coerce').fillna(0).astype(int)
        gender_tensor = torch.LongTensor(gender_values)
    else:
        gender_tensor = torch.LongTensor(np.zeros(len(X_train), dtype=int))
    
    # Create simple batch info (just zeros for now)
    info_tensor = torch.FloatTensor(np.zeros((len(X_train), 1)))
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, gender_tensor, info_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.3
    )
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, (batch_x, batch_gender, batch_info, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_gender = batch_gender.to(device)
            batch_info = batch_info.to(device)
            batch_y = batch_y.to(device)
            
            # Compute training length ratio for adaptive loss weighting
            training_length_ratio = 0.0 if epoch < 10 else (epoch - 10) / (epochs - 10)
            
            optimizer.zero_grad()
            
            # Forward pass and compute loss
            losses = model.compute_loss(batch_x, batch_gender, batch_info, batch_y, batch_y, training_length_ratio)
            loss = losses['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Memory cleanup
        if epoch % 10 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return model
