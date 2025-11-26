import sys
import anndata as ad
# ... dependencies

CODE_PATH="/opt/OpenProblems2022Analysis/code/rank1/open-problems-multimodal"

## VIASH START
par = {
    'input_train_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/train_mod1.h5ad',
    'input_train_mod2': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/train_mod2.h5ad',
    'input_test_mod1': 'resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/test_mod1.h5ad',
    'output': 'output.h5ad'

}
meta = {
    'name': 'ss_opm',
    # this is available, after running viash ns build
    'resources_dir': 'target/executable/methods/ss_opm',
}
CODE_PATH="/path/to/local/repo/OpenProblems2022Analysis/code/rank1/open-problems-multimodal"
## VIASH END

# sys.path.append(meta['resources_dir'])
# from utils import get_representation

# sys.path.append(CODE_PATH)

print('Reading input files', flush=True)
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

# TODO: fill in
mod2_pred = None

print("Write output AnnData to file", flush=True)
output = ad.AnnData(
    layers={"normalized": mod2_pred},
    obs=input_test_mod1.obs[[]],
    var=input_train_mod2.var[[]],
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': meta["name"],
    },
)
output.write_h5ad(par['output'], compression='gzip')
