#!/bin/bash
#SBATCH --job-name=AUC_prompting_job  # Specify a name for your job
#SBATCH --output=outputs/AUC_trial.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/errors_AUC_trial.log # AUC_p_tuning_main_errors.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --time=10:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --mem=32G                  # Memory per node (4GB in this example)

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

# Navigate to the directory containing your Python code
# cd /nfshomes/qhe123/AUC

# Activate your base environment (replace "your_base_env" with the name of your environment)
# conda init bash
# conda activate prompting

# Execute your Python code
python3 main_AUC_trainer.py
# Deactivate the environment (if you want to)
# conda deactivate

# Your job is done!

