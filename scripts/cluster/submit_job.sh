#!/bin/bash
#$ -cwd
#$ -j y
#$ -l gpu,A100,cuda=1,h_data=32G,h_rt=4:00:00
#$ -N cross_species_transfer
#$ -M your.email@example.com
#$ -m bea

# UCLA UGE Cluster Job Script for Cross-Species Label Transfer
# Modify the email address above and resource requirements as needed

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load modules
module load anaconda3
module load gcc/10.2.0
module load cmake/3.30.0

# Activate conda environment
source ~/.bashrc
conda activate scgpt

# Display environment info
echo ""
echo "Environment Information:"
echo "----------------------------------------"
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "GPU device:"
    python -c "import torch; print(torch.cuda.get_device_name(0))"
fi
echo "----------------------------------------"
echo ""

# Parse command line arguments
CONFIG_FILE=${1:-configs/experiments/mouse_to_opossum.yaml}
GPU_ID=${2:-0}

echo "Configuration file: $CONFIG_FILE"
echo "GPU ID: $GPU_ID"
echo ""

# Run the experiment
echo "Starting experiment..."
echo "=========================================="
python scripts/run_experiment.py \
    --config "$CONFIG_FILE" \
    --gpu "$GPU_ID"

EXIT_CODE=$?

echo "=========================================="
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE