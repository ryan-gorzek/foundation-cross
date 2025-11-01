#!/usr/bin/env Rscript
# Convert AnnData H5AD files to Seurat objects

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
  library(optparse)
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

# Convert H5AD to H5Seurat (intermediate format)
h5seurat_path <- sub("\\.h5ad$", ".h5seurat", opt$input)
Convert(opt$input, dest = "h5seurat", overwrite = TRUE)

# Load as Seurat object
seurat_obj <- LoadH5Seurat(h5seurat_path, assays = opt$assay)

# Clean up intermediate file
if (file.exists(h5seurat_path)) {
  file.remove(h5seurat_path)
}

# Save as RDS
saveRDS(seurat_obj, file = opt$output)

cat("Conversion complete!\n")
cat("Seurat object saved to:", opt$output, "\n")
cat("Cells:", ncol(seurat_obj), "\n")
cat("Genes:", nrow(seurat_obj), "\n")