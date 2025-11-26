
Rebuild docker image

```bash
viash run src/methods/ss_opm/config.vsh.yaml -- ---setup cb ---verbose
```

View docker image

```bash
viash run src/methods/ss_opm/config.vsh.yaml -- ---dockerfile
```

Get command to run docker image

```bash
viash run src/methods/ss_opm/config.vsh.yaml -- ---debug
```

Run tests

```bash
viash test src/methods/ss_opm/config.vsh.yaml
```

Run method manually on data:

```bash
DIR=resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/swap/

viash run src/methods/ss_opm/config.vsh.yaml -- \
  --input_train_mod1 "$DIR/train_mod1.h5ad" \
  --input_train_mod2 "$DIR/train_mod2.h5ad" \
  --input_test_mod1 "$DIR/test_mod1.h5ad" \
  --output output.h5ad
```
