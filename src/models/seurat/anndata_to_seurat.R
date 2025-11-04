#!/usr/bin/env Rscript
# Convert AnnData H5AD files to Seurat objects

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
  library(optparse)
  library(rhdf5)
})

# Parse command line arguments
option_list <- list(
  make_option(c("--input"), type="character", default=NULL,
              help="Input H5AD file path", metavar="character"),
  make_option(c("--output"), type="character", default=NULL,
              help="Output RDS file path", metavar="character"),
  make_option(c("--assay"), type="character", default="RNA",
              help="Assay name [default= %default]", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$input) || is.null(opt$output)) {
  print_help(opt_parser)
  stop("Both --input and --output must be specified.", call.=FALSE)
}

cat("Converting H5AD to Seurat object...\n")
cat("Input:", opt$input, "\n")
cat("Output:", opt$output, "\n")

# Set Seurat to v5 assay format
options(Seurat.object.assay.version = "v5")

# Convert H5AD to H5Seurat (intermediate format)
h5seurat_path <- sub("\\.h5ad$", ".h5seurat", opt$input)
Convert(opt$input, dest = "h5seurat", overwrite = TRUE)

# Load as Seurat object (without metadata - will load manually)
seurat_obj <- LoadH5Seurat(h5seurat_path, assays = opt$assay, meta.data = FALSE)

cat("Loading metadata from H5AD...\n")

# Read obs (metadata) directly from H5AD file
obs <- h5read(opt$input, "/obs")

# Read cell barcodes from H5AD
h5_obs_names <- h5read(opt$input, "/obs/_index")

# Get cell barcodes from Seurat object
seurat_cell_names <- colnames(seurat_obj)

cat("H5AD obs rows:", length(h5_obs_names), "\n")
cat("Seurat cells:", length(seurat_cell_names), "\n")

# Check column lengths
cat("\nValidating obs column lengths...\n")
for (col_name in names(obs)) {
  col_data <- obs[[col_name]]
  if (is.list(col_data) && "codes" %in% names(col_data)) {
    col_len <- length(col_data$codes)
  } else {
    col_len <- length(col_data)
  }
  
  if (col_len != length(h5_obs_names)) {
    cat("WARNING:", col_name, "has", col_len, "values but expected", length(h5_obs_names), "\n")
  }
}

# Find matching indices
match_indices <- match(seurat_cell_names, h5_obs_names)

if (any(is.na(match_indices))) {
  cat("WARNING:", sum(is.na(match_indices)), "cells in Seurat not found in H5AD obs\n")
}

# Convert categorical columns to character vectors
obs_df <- data.frame(row.names = seurat_cell_names)

for (col_name in names(obs)) {
  col_data <- obs[[col_name]]
  
  # Check if this is categorical (has categories and codes)
  if (is.list(col_data) && "categories" %in% names(col_data) && "codes" %in% names(col_data)) {
    # Categorical column: map codes to categories
    categories <- col_data$categories
    codes <- col_data$codes
    
    # CRITICAL: Force codes to be a simple vector (may have dimensions from HDF5)
    if (!is.null(dim(codes))) {
      cat("  ", col_name, ": codes has dimensions", dim(codes), "- converting to vector\n")
      codes <- as.vector(codes)
    }
    
    # R uses 1-based indexing, codes are 0-based
    # Extract codes for matched cells
    matched_codes <- codes[match_indices]
    
    # Handle -1 (missing/NA values in categorical)
    # -1 in codes means NA, don't try to index with it
    values <- rep(NA_character_, length(matched_codes))
    valid_mask <- matched_codes >= 0
    
    if (any(valid_mask)) {
      values[valid_mask] <- categories[matched_codes[valid_mask] + 1]
    }
    
    if (any(!valid_mask)) {
      cat("  ", col_name, ": has", sum(!valid_mask), "NA values\n")
    }
    
    # Convert to factor
    obs_df[[col_name]] <- factor(values, levels = categories)
    
    cat("  ", col_name, ": categorical with", length(categories), "levels\n")
  } else {
    # Non-categorical column: use matched indices
    obs_df[[col_name]] <- col_data[match_indices]
    cat("  ", col_name, ": numeric/other\n")
  }
}

# Assign metadata to Seurat object
seurat_obj@meta.data <- obs_df

cat("Metadata loaded:", ncol(obs_df), "columns\n")

# Clean up intermediate file
if (file.exists(h5seurat_path)) {
  file.remove(h5seurat_path)
}

# Ensure object contains Assay5 and save as RDS
seurat_obj[["RNA"]] <- as(object = seurat_obj[["RNA"]], Class = "Assay5")
saveRDS(seurat_obj, file = opt$output)

cat("Conversion complete!\n")
cat("Seurat object saved to:", opt$output, "\n")
cat("Cells:", ncol(seurat_obj), "\n")
cat("Genes:", nrow(seurat_obj), "\n")
cat("Metadata columns:", paste(colnames(seurat_obj@meta.data), collapse=", "), "\n")