cat("Loading dependencies\n")
requireNamespace("anndata", quietly = TRUE)

## VIASH START
par <- list(
  input_test_mod2 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/test_mod2.h5ad",
  output = "output.h5ad"
)

meta <- list(
  name = "foo"
)
## VIASH END

cat("Reading h5ad files\n")
ad2_test <- anndata::read_h5ad(par$input_test_mod2)
ad2_test$uns[["method_id"]] <- meta$name

cat("Writing predictions to file\n")
zzz <- ad2_test$write_h5ad(par$output, compression = "gzip")
