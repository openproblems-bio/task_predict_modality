import numpy as np
import yaml
from collections import namedtuple


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

    Xt = tr1[mask].layers["normalized"].toarray()
    X = tr1[maskr].layers["normalized"].toarray()

    yt = tr2[mask].layers["normalized"].toarray()
    y = tr2[maskr].layers["normalized"].toarray()

    print(f"{X.shape}, {y.shape}, {Xt.shape}, {yt.shape}")

    return X,y,Xt,yt


def load_yaml(path):
    with open(path) as f:
        x = yaml.safe_load(f)
    res = {}
    for i in x:
        res[i] = x[i]['value']
    config = namedtuple('Config', res.keys())(**res)
    print(config)
    return config
