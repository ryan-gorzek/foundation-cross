#!/usr/bin/env Rscript
# Seurat MapQuery for cross-species label transfer

suppressPackageStartupMessages({
  library(Seurat)
  library(optparse)
  library(jsonlite)
})

# Parse command line arguments
option_list <- list(
  make_option(c("--reference"), type="character", default=NULL,
              help="Reference Seurat RDS file path", metavar="character"),
  make_option(c("--query"), type="character", default=NULL,
              help="Query Seurat RDS file path", metavar="character"),
  make_option(c("--output"), type="character", default=NULL,
              help="Output directory", metavar="character"),
  make_option(c("--config"), type="character", default=NULL,
              help="Configuration JSON file", metavar="character"),
  make_option(c("--celltype_column"), type="character", default="celltype",
              help="Cell type column name [default= %default]", metavar="character"),
  make_option(c("--dims"), type="integer", default=30,
              help="Number of dimensions for PCA [default= %default]", metavar="integer"),
  make_option(c("--k_anchor"), type="integer", default=5,
              help="Number of neighbors for anchors [default= %default]", metavar="integer"),
  make_option(c("--k_filter"), type="integer", default=200,
              help="Number of neighbors for filtering [default= %default]", metavar="integer"),
  make_option(c("--k_weight"), type="integer", default=50,
              help="Number of neighbors for weight [default= %default]", metavar="integer")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Load config if provided
if (!is.null(opt$config) && file.exists(opt$config)) {
  config <- fromJSON(opt$config)
  # Override with config values
  if (!is.null(config$celltype_column)) opt$celltype_column <- config$celltype_column
  if (!is.null(config$dims)) opt$dims <- config$dims
  if (!is.null(config$k_anchor)) opt$k_anchor <- config$k_anchor
  if (!is.null(config$k_filter)) opt$k_filter <- config$k_filter
  if (!is.null(config$k_weight)) opt$k_weight <- config$k_weight
}

if (is.null(opt$reference) || is.null(opt$query) || is.null(opt$output)) {
  print_help(opt_parser)
  stop("--reference, --query, and --output must be specified.", call.=FALSE)
}

cat("=======================================================\n")
cat("Seurat MapQuery Label Transfer\n")
cat("=======================================================\n")

# Create output directory
dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)

# Load data
cat("\nLoading reference data...\n")
reference <- readRDS(opt$reference)
cat("Reference:", ncol(reference), "cells,", nrow(reference), "genes\n")

cat("\nLoading query data...\n")
query <- readRDS(opt$query)
cat("Query:", ncol(query), "cells,", nrow(query), "genes\n")

# Check for celltype column in reference
if (!(opt$celltype_column %in% colnames(reference@meta.data))) {
  stop(paste("Cell type column", opt$celltype_column, "not found in reference metadata."), call.=FALSE)
}

# Preprocessing reference
cat("\nPreprocessing reference...\n")

# Handle Seurat v5 assays - join layers if needed
if (inherits(reference[["RNA"]], "Assay5")) {
  cat("Detected Seurat v5 assay - joining layers...\n")
  reference[["RNA"]] <- JoinLayers(reference[["RNA"]])
}

reference <- NormalizeData(reference, verbose = FALSE)
reference <- FindVariableFeatures(reference, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
reference <- ScaleData(reference, verbose = FALSE)
reference <- RunPCA(reference, npcs = opt$dims, verbose = FALSE)

# Preprocessing query
cat("Preprocessing query...\n")

# Handle Seurat v5 assays - join layers if needed
if (inherits(query[["RNA"]], "Assay5")) {
  cat("Detected Seurat v5 assay - joining layers...\n")
  query[["RNA"]] <- JoinLayers(query[["RNA"]])
}

query <- NormalizeData(query, verbose = FALSE)

# Find transfer anchors
cat("\nFinding transfer anchors...\n")
cat("Parameters:\n")
cat("  dims:", opt$dims, "\n")
cat("  k.anchor:", opt$k_anchor, "\n")
cat("  k.filter:", opt$k_filter, "\n")

anchors <- FindTransferAnchors(
  reference = reference,
  query = query,
  dims = 1:opt$dims,
  k.anchor = opt$k_anchor,
  k.filter = opt$k_filter,
  verbose = TRUE
)

cat("\nFound", nrow(anchors@anchors), "anchors\n")

# Transfer labels
cat("\nTransferring labels...\n")
predictions <- TransferData(
  anchorset = anchors,
  refdata = reference@meta.data[[opt$celltype_column]],
  dims = 1:opt$dims,
  k.weight = opt$k_weight,
  verbose = TRUE
)

# Extract predictions
predicted_labels <- predictions$predicted.id
prediction_scores <- predictions$prediction.score.max

cat("\nTransfer complete!\n")
cat("Predicted", length(predicted_labels), "cell labels\n")

# Save predictions to CSV
predictions_df <- data.frame(
  cell_id = colnames(query),
  predicted_label = predicted_labels,
  prediction_score = prediction_scores,
  stringsAsFactors = FALSE
)

# Add true labels if available
if (opt$celltype_column %in% colnames(query@meta.data)) {
  predictions_df$true_label <- query@meta.data[[opt$celltype_column]]
}

output_csv <- file.path(opt$output, "predictions.csv")
write.csv(predictions_df, file = output_csv, row.names = FALSE, quote = FALSE)
cat("Predictions saved to:", output_csv, "\n")

# Save query with predictions
query@meta.data$predicted_celltype <- predicted_labels
query@meta.data$prediction_score <- prediction_scores

output_query <- file.path(opt$output, "query_with_predictions.rds")
saveRDS(query, file = output_query)
cat("Query object with predictions saved to:", output_query, "\n")

# Save metadata for Python
metadata <- list(
  n_reference_cells = ncol(reference),
  n_query_cells = ncol(query),
  n_anchors = nrow(anchors@anchors),
  celltype_column = opt$celltype_column,
  reference_celltypes = unique(as.character(reference@meta.data[[opt$celltype_column]])),
  dims = opt$dims,
  k_anchor = opt$k_anchor,
  k_filter = opt$k_filter,
  k_weight = opt$k_weight
)

metadata_json <- file.path(opt$output, "transfer_metadata.json")
write(toJSON(metadata, pretty = TRUE, auto_unbox = TRUE), file = metadata_json)
cat("Metadata saved to:", metadata_json, "\n")

cat("\n=======================================================\n")
cat("MapQuery complete!\n")
cat("=======================================================\n")