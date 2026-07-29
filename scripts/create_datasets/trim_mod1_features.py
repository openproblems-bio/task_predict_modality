#!/usr/bin/env python3
"""Trim mod1 to fewer features than mod2 in the generated test resources.

The common test datasets happen to be square -- bmmc_cite is 134 x 134 and
bmmc_multiome 1500 x 1500 -- so in resources_test a component can confuse
n_vars(mod1) with n_vars(mod2) and still pass `viash test`. That is not true of
any real dataset, where e.g. GEX has ~13953 features against 134 ADT proteins.

Dropping a slice of mod1's features restores the asymmetry, so dimension
mix-ups fail in CI rather than on the first full run.
"""

import sys
from pathlib import Path

import anndata as ad

# keep this fraction of mod1's features
KEEP_FRACTION = 0.75


def trim(path: Path) -> None:
    adata = ad.read_h5ad(path)
    n_keep = int(adata.n_vars * KEEP_FRACTION)
    trimmed = adata[:, :n_keep].copy()
    trimmed.write_h5ad(path, compression="gzip")
    print(f"  {path}: {adata.n_vars} -> {trimmed.n_vars} features", flush=True)


def main(dataset_dir: str) -> None:
    for name in ["train_mod1.h5ad", "test_mod1.h5ad"]:
        trim(Path(dataset_dir) / name)


if __name__ == "__main__":
    main(sys.argv[1])
