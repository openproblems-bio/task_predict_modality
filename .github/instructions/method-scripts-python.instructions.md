---
description: "Use when writing, fixing, or reviewing method/metric script.py files in src/methods/, src/metrics/, or src/control_methods/. Covers script style, API compliance, and how to verify components."
applyTo: "src/methods/**/script.py,src/metrics/**/script.py,src/control_methods/**/script.py"
---
# Method & Metric Script Guidelines (Python)

## Core Principle

`script.py` should represent **typical bioinformatician usage** of the tool with minimal modifications. Only adapt what is strictly necessary to:
1. Read inputs from the paths provided by `par`
2. Pass the right data structures to the method
3. Convert the method's output back into the expected output structures
4. Write outputs to `par['output']`

Do **not** restructure the method's native API, add abstraction layers, or rewrite the algorithm logic.

## Finding API Specs

Input/output file formats are defined in `src/api/`. Key files:
- `file_train_mod1.yaml` / `file_train_mod2.yaml` — training data AnnData fields (mod1 and mod2)
- `file_test_mod1.yaml` / `file_test_mod2.yaml` — test data AnnData fields
- `file_prediction.yaml` — expected output format for methods (has `layers["normalized"]`)
- `file_pretrained_model.yaml` — expected output format for train components
- `file_score.yaml` — expected output format for metrics
- `comp_method.yaml`, `comp_metric.yaml` — component argument specs

Always check these before deciding what fields to read or write.

## The `## VIASH START` / `## VIASH END` Block

This block is **auto-generated** by viash from the component's `config.vsh.yaml` arguments. It is replaced at build/test time with a real CLI parser. Keep it in the script only as a local development convenience.

- **Do not edit it manually** to add or remove parameters — edit `config.vsh.yaml` instead.
- After adding, removing, or renaming an argument in the config, regenerate the block:
  ```bash
  viash config inject src/methods/<name>/config.vsh.yaml
  ```
- Argument names in the config (`--my_param`) map directly to `par['my_param']` keys.

## Common Patterns

**Reading inputs (single-step method):**
```python
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
```

**Writing prediction output:**
```python
output = ad.AnnData(
    layers={"normalized": y_pred},
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        "dataset_id": input_train_mod1.uns["dataset_id"],
        "method_id": meta["name"],
    },
)
output.write_h5ad(par['output'], compression="gzip")
```

**Writing metric score output:**
```python
output = ad.AnnData(
    uns={
        "dataset_id": ad_pred.uns["dataset_id"],
        "method_id": ad_pred.uns["method_id"],
        "metric_ids": ["my_metric"],
        "metric_values": [score],
    }
)
output.write_h5ad(par['output'], compression="gzip")
```

**Reading inputs (predict component of a two-step method):**
```python
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
# load model from par['input_model'] directory
```

## Dependency Fixes

If a library has a dependency conflict (e.g., incompatible with newer `anndata`, `numpy`, etc.), prefer replacing it with an alternative that provides the same model/algorithm natively rather than pinning transitive dependencies.

Update `config.vsh.yaml` to remove the broken package from the `setup` block when replacing it.

## Verification

After any change to a method script or config, verify with:
```bash
viash test src/methods/<name>/config.vsh.yaml
# or
viash test src/metrics/<name>/config.vsh.yaml
```

Both test scripts must succeed (`2 out of 2`).
