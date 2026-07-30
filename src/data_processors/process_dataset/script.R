cat("Loading dependencies\n")
library(anndata, warn.conflicts = FALSE)
library(Matrix, warn.conflicts = FALSE)

## VIASH START
par <- list(
  input_mod1 = "resources_test/common/openproblems_neurips2021/bmmc_cite/dataset_mod1.h5ad",
  input_mod2 = "resources_test/common/openproblems_neurips2021/bmmc_cite/dataset_mod2.h5ad",
  output_train_mod1 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/train_mod1.h5ad",
  output_train_mod2 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/train_mod2.h5ad",
  output_test_mod1 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/test_mod1.h5ad",
  output_test_mod2 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_cite/test_mod2.h5ad",
  swap = TRUE,
  seed = 1L
)
# par <- list(
#   input_mod1 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/output_mod1.h5ad",
#   input_mod2 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/output_atac.h5ad",
#   output_train_mod1 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/train_mod1.h5ad",
#   output_train_mod2 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/train_mod2.h5ad",
#   output_test_mod1 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/test_mod1.h5ad",
#   output_test_mod2 = "resources_test/task_predict_modality/openproblems_neurips2021/bmmc_multiome/test_mod2.h5ad",
#   swap = TRUE,
#   seed = 1L
# )
## VIASH END

cat("Using seed ", par$seed, "\n", sep = "")
set.seed(par$seed)

cat("Reading input data\n")
ad1 <- anndata::read_h5ad(if (!par$swap) par$input_mod1 else par$input_mod2)
ad2 <- anndata::read_h5ad(if (!par$swap) par$input_mod2 else par$input_mod1)

# input checks -- used to balance the subsample of test cells further down
if (!"cell_type" %in% colnames(ad1$obs)) {
  stop("obs['cell_type'] is required but missing from the input dataset")
}

# use heuristic to determine modality
# TODO: should be removed once modality is stored in the uns
determine_modality <- function(ad, mod1 = TRUE) {
  if ("modality" %in% names(ad$uns)) {
    ad$uns[["modality"]]
  } else if ("feature_types" %in% colnames(ad$var)) {
    unique(ad$var[["feature_types"]])
  } else if (mod1) {
    "GEX"
  } else if (grepl("cite", ad$uns[["dataset_id"]])) {
    "ADT"
  } else if (grepl("multiome", ad$uns[["dataset_id"]])) {
    "ATAC"
  } else {
    stop("Could not determine modality")
  }
}
ad1_mod <- determine_modality(ad1, !par$swap)
ad2_mod <- determine_modality(ad2, par$swap)

# determine new uns
uns_vars <- c("dataset_id", "dataset_name", "dataset_url", "dataset_reference", "dataset_summary", "dataset_description", "dataset_organism", "normalization_id")
ad1_uns <- ad1$uns[uns_vars]
ad2_uns <- ad2$uns[uns_vars]
ad1_uns$modality <- ad1_mod
ad2_uns$modality <- ad2_mod

# Create new dataset id and name depending on the modality
if (!is.null(par$dataset_id)) {
  ad1_uns[["common_dataset_id"]] <- ad2_uns[["common_dataset_id"]] <- ad1_uns$dataset_id
  ad1_uns$dataset_id <- ad2_uns$dataset_id <- par$dataset_id
}

new_dataset_name <- paste0(ad1_uns$dataset_name, " (", ad1_mod, "2", ad2_mod, ")")
ad1_uns$dataset_name <- ad2_uns$dataset_name <- new_dataset_name

# determine new obsm
ad1_obsm <- ad2_obsm <- list()

# determine new var
ad1_var <- ad1$var[, intersect(colnames(ad1$var), c("gene_ids", "hvg", "hvg_score")), drop = FALSE]
ad2_var <- ad2$var[, intersect(colnames(ad2$var), c("gene_ids", "hvg", "hvg_score")), drop = FALSE]

if (ad1_mod == "ATAC" && "gene_activity" %in% names(ad1$obsm)) {
  # copy gene activity in new object
  ad1_uns$gene_activity_var_names <- ad1$uns$gene_activity_var_names
  ad1_obsm$gene_activity <- as(ad1$obsm$gene_activity, "CsparseMatrix")
}

if (ad2_mod == "ATAC") {
  # subset to make the task computationally feasible
  if (ncol(ad2) > 10000) {
    poss_ix <- which(Matrix::colSums(ad2$layers[["normalized"]]) > 0)
    # sample by position -- sample(x, n) errors when n > length(x), and treats a
    # length-1 x as seq_len(x)
    sel_ix <- sort(poss_ix[sample.int(length(poss_ix), min(10000, length(poss_ix)))])
    ad2 <- ad2[, sel_ix]$copy()
    ad2_var <- ad2_var[sel_ix, , drop = FALSE]
  }

  if ("gene_activity" %in% names(ad2$obsm)) {
    # copy gene activity in new object
    ad2_uns$gene_activity_var_names <- ad2$uns$gene_activity_var_names
    ad2_obsm$gene_activity <- as(ad2$obsm$gene_activity, "CsparseMatrix")
  }
}

cat("Creating train/test split\n")
if ("is_train" %in% colnames(ad1$obs)) {
  is_train <- which(ad1$obs[["is_train"]] == "train")
  is_test <- which(ad1$obs[["is_train"]] != "train")
} else {
  # No predefined split -- obs['is_train'] carries the NeurIPS 2021 competition
  # split and other datasets have no reason to have it. Hold out a quarter of the
  # batches instead, so the test cells come from donors the method has not seen.
  batches <- unique(as.character(ad1$obs[["batch"]]))
  if (length(batches) > 1) {
    test_batches <- sample(batches, max(1, floor(length(batches) / 4)))
    cat("No obs['is_train'], holding out batches: ", paste(test_batches, collapse = ", "), "\n", sep = "")
    in_test <- as.character(ad1$obs[["batch"]]) %in% test_batches
  } else {
    cat("No obs['is_train'] and only one batch, holding out a quarter of the cells\n")
    in_test <- seq_len(nrow(ad1)) %in% sample.int(nrow(ad1), max(1, floor(nrow(ad1) / 4)))
  }
  is_train <- which(!in_test)
  is_test <- which(in_test)
}
cat("Train cells: ", length(is_train), ", test cells: ", length(is_test), "\n", sep = "")

# sample cells
if (length(is_test) > 1000) {
  ct <- as.character(ad1$obs[["cell_type"]][is_test])
  ct_tab <- table(ct)
  ct_freq <- setNames(as.vector(ct_tab) / sum(ct_tab), names(ct_tab))
  is_test <- sample(is_test, 1000, prob = sqrt(1 / ct_freq[ct]))
}

train_obs <- ad1$obs[is_train, intersect(colnames(ad1$obs), c("batch", "size_factors")), drop = FALSE]
test_obs <- ad1$obs[is_test, intersect(colnames(ad1$obs), c("batch", "size_factors")), drop = FALSE]
subset_mats <- function(li, obs_filt) {
  out <- list()
  for (n in names(li)) {
    out[[n]] <- li[[n]][obs_filt, , drop = FALSE]
  }
  out
}

cat("Create train objects\n")
output_train_mod1 <- anndata::AnnData(
  layers = subset_mats(list(counts = ad1$layers[["counts"]], normalized = ad1$layers[["normalized"]]), is_train),
  obsm = subset_mats(ad1_obsm, is_train),
  obs = train_obs,
  var = ad1_var,
  uns = ad1_uns
)
output_train_mod2 <- anndata::AnnData(
  layers = subset_mats(list(counts = ad2$layers[["counts"]], normalized = ad2$layers[["normalized"]]), is_train),
  obsm = subset_mats(ad2_obsm, is_train),
  obs = train_obs,
  var = ad2_var,
  uns = ad2_uns
)

cat("Create test objects\n")
output_test_mod1 <- anndata::AnnData(
  layers = subset_mats(list(counts = ad1$layers[["counts"]], normalized = ad1$layers[["normalized"]]), is_test),
  obsm = subset_mats(ad1_obsm, is_test),
  obs = test_obs,
  var = ad1_var,
  uns = ad1_uns
)
output_test_mod2 <- anndata::AnnData(
  layers = subset_mats(list(counts = ad2$layers[["counts"]], normalized = ad2$layers[["normalized"]]), is_test),
  obsm = subset_mats(ad2_obsm, is_test),
  obs = test_obs,
  var = ad2_var,
  uns = ad2_uns
)

cat("Saving output files as h5ad\n")
zzz <- output_train_mod1$write_h5ad(par$output_train_mod1, compression = "gzip")
zzz <- output_train_mod2$write_h5ad(par$output_train_mod2, compression = "gzip")
zzz <- output_test_mod1$write_h5ad(par$output_test_mod1, compression = "gzip")
zzz <- output_test_mod2$write_h5ad(par$output_test_mod2, compression = "gzip")
