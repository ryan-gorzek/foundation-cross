#!/bin/bash
#$ -cwd
#$ -j y
#$ -l gpu,A100,cuda=1,h_data=32G,h_rt=2:00:00
#$ -N scgpt_train
#$ -M rgorzek@ucla.edu
#$ -m bea

# UCLA UGE Cluster Job Script for scGPT Training
# Modify the email address above

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"
echo "Working directory: $(pwd)"

# Activate conda environment
# Replace 'scgpt' with your actual environment name
source ~/.bashrc
module load anaconda3
conda activate scgpt_env

# Display environment info
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

# Run the training pipeline
echo "Starting training..."
python scripts/scgpt_pipeline/train.py

echo "Job completed at $(date)"
