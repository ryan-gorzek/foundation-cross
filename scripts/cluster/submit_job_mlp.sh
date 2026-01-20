#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
#$ -o jobs/joblog.$JOB_ID
#$ -j y
#$ -l gpu,A6000,cuda=1,h_data=32G,h_rt=00:10:00
#$ -N cross_species_transfer
#$ -M rgorzek@ucla.edu
#$ -m bea

# UCLA Hoffman2 Cluster Job Script for Cross-Species Label Transfer
# Modify the email address above and resource requirements as needed

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load the job environment
. /u/local/Modules/default/init/modules.sh
# Set up environment
module load anaconda3
conda activate scgpt_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

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
CONFIG_FILE=${1:-configs/experiments/mouse_to_opossum_mlp.yaml}

echo "Configuration file: $CONFIG_FILE"
echo ""

# Run the experiment
echo "Starting experiment..."
echo "=========================================="
python scripts/run_experiment.py \
    --config "$CONFIG_FILE"

# echo job info on joblog:
echo "=========================================="
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
echo "=========================================="
#### submit_job.sh STOP ####
