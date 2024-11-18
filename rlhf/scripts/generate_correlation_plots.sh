#!/bin/bash

# Set the experiment parameters
#envs=("Ant-v5" "Hopper-v5" "Humanoid-v5")
envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5")
seeds=(1789 1687123 12)

# Create a directory for log files if it doesn't exist
mkdir -p logs

# Prepare an array to hold all combinations
declare -a combinations

# Generate all combinations
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        combinations+=("$seed $env")
    done
done

# Set the batch size (number of jobs per GPU)
batch_size=4
total_combinations=${#combinations[@]}

# Loop over the combinations in batches
for ((i=0; i<$total_combinations; i+=$batch_size)); do
    batch=("${combinations[@]:$i:$batch_size}")
    batch_id=$((i / batch_size))

    # Create a temporary Slurm job script for this batch
    sbatch_script="batch_job_$batch_id.sh"
    cat <<EOT > $sbatch_script
#!/bin/bash
#SBATCH --partition=gpu_4,gpu_8,gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=generate_feedback_$batch_id
#SBATCH --time=01:30:00
#SBATCH --output=logs/generate_corr_plots_${batch_id}_%j.out

# Load any necessary modules or activate environments here
# module load python/3.9
source /pfs/data5/home/kn/kn_kn/kn_pop257914/multi-type-feedback/venv/bin/activate

# Run the training jobs in background
EOT

    # Add each task to the Slurm script
    for combination in "${batch[@]}"; do
        read seed env <<< $combination
        echo "python generate_rew_correlation_plot.py --algorithm ppo --environment $env --seed $seed --n-feedback 10000 --noise-level 0.75 &" >> $sbatch_script
    done

    # Wait for all background jobs to finish
    echo "wait" >> $sbatch_script

    # Submit the Slurm job script
    sbatch $sbatch_script

    # Optional: Remove the temporary Slurm script
    # rm $sbatch_script
done

echo "All jobs have been submitted."
