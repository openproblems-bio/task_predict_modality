import numpy as np
import scipy.sparse
import torch
import yaml
from collections import namedtuple
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler


def to_site_donor(data):
    df = data.obs['batch'].copy().to_frame().reset_index()
    df.columns = ['index','batch']
    df['site'] = df['batch'].apply(lambda x: x[:2])
    df['donor'] = df['batch'].apply(lambda x: x[2:]) 
    return df


def split(tr1, tr2, fold):
    df = to_site_donor(tr1)
    mask = (df['site'] == f's{fold+1}').values

    # batch labels outside the NeurIPS 2021 s{site}d{donor} naming match no site, which
    # would leave the validation half empty; hold out every third batch instead
    if not mask.any():
        batches = sorted(df['batch'].unique())
        if len(batches) >= 3:
            mask = df['batch'].isin(batches[fold::3]).values
        else:
            mask = np.arange(len(df)) % 3 == fold

    maskr = ~mask

    # kept sparse: densified, the NeurIPS 2022 ATAC input alone needs 143 GB; see batch_loader
    Xt = float32_rows(tr1.layers["normalized"], mask)
    X = float32_rows(tr1.layers["normalized"], maskr)

    yt = float32_rows(tr2.layers["normalized"], mask)
    y = float32_rows(tr2.layers["normalized"], maskr)

    print(f"{X.shape}, {y.shape}, {Xt.shape}, {yt.shape}")

    return X,y,Xt,yt


def float32_rows(matrix, rows):
    """The selected rows of a sparse matrix as float32 CSR, the precision the MLP trains in."""
    return scipy.sparse.csr_matrix(matrix[rows], dtype=np.float32)


class SparseRows(Dataset):
    """Rows of one or more sparse matrices, served as dense float32 tensors a whole batch at a time."""

    def __init__(self, *matrices):
        self.matrices = [scipy.sparse.csr_matrix(matrix, dtype=np.float32) for matrix in matrices]

    def __len__(self):
        return self.matrices[0].shape[0]

    def __getitem__(self, rows):
        return tuple(torch.from_numpy(matrix[rows].toarray()) for matrix in self.matrices)


def batch_loader(dataset, batch_size, shuffle, drop_last, num_workers):
    """DataLoader that hands `dataset` the row indices of a whole batch, so only one batch is densified at a time."""
    rows = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=BatchSampler(rows, batch_size=batch_size, drop_last=drop_last),
        batch_size=None,
        num_workers=num_workers,
    )


def load_yaml(path):
    with open(path) as f:
        x = yaml.safe_load(f)
    res = {}
    for i in x:
        res[i] = x[i]['value']
    config = namedtuple('Config', res.keys())(**res)
    print(config)
    return config
