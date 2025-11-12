#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
#$ -o jobs/joblog.$JOB_ID
#$ -j y
#$ -l h_data=32G,h_rt=5:00:00
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

# Display environment info
echo ""
echo "Environment Information:"
echo "----------------------------------------"
echo "Python version:"
python --version
echo ""
echo "----------------------------------------"
echo ""

# Parse command line arguments
CONFIG_FILE=${1:-configs/experiments/mouse_to_opossum_scvi.yaml}

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
