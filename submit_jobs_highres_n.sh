#!/bin/bash
#
# Submit jobs for high-resolution n sweep (for nice value vs n figures)
# Uses fewer H values but many more n values
#
# Usage: ./submit_jobs_highres_n.sh [--dry-run]
#

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="/sci/labs/orzuk/orzuk/github/BandedDecomposition"
CONDA_ENV="pymol-env"

# High-resolution n values (100 to 2000, step 100)
N_VALUES=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)

# Fewer H values (well-conditioned range for reliable results)
H_VALUES=(0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90)

# Alpha values
ALPHA_VALUES=(1.0 5.0)

# Only mixed_fbm model
MODEL="mixed_fbm"

# All strategies
STRATEGIES=("markovian" "full")

# Check for dry run
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# Create jobs directory
JOBS_DIR="${SCRIPT_DIR}/slurm_jobs_highres"
mkdir -p "$JOBS_DIR"

job_count=0

echo "=============================================="
echo "High-resolution n sweep for value vs n figures"
echo "=============================================="
echo ""
echo "Model: ${MODEL}"
echo "N values: ${N_VALUES[*]}"
echo "H values: ${H_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Strategies: ${STRATEGIES[*]}"
echo ""

for alpha in "${ALPHA_VALUES[@]}"; do
    for H in "${H_VALUES[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            # SLURM settings based on strategy (increased for n up to 2000)
            if [ "$strategy" == "markovian" ]; then
                time="08:00:00"
                mem="8G"
                METHOD="precond-newton-cg"
            else
                # Full strategy is slower and needs more memory for large n
                time="24:00:00"
                mem="32G"
                METHOD="newton-cg"  # No preconditioning for full
            fi
            cpus="4"

            job_name="highres_a${alpha}_H${H}_${strategy}"
            job_script="${JOBS_DIR}/${job_name}.sh"

            # Create SLURM job script
            cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${JOBS_DIR}/${job_name}_%j.out
#SBATCH --error=${JOBS_DIR}/${job_name}_%j.err
#SBATCH --time=${time}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}

cd ${WORK_DIR}

# Initialize conda
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

echo "Starting high-res n sweep: alpha=${alpha}, H=${H}, strategy=${strategy}"
echo "N values: ${N_VALUES[*]}"
echo "Method: ${METHOD}"
echo "Time: \$(date)"
echo ""

# Loop over n values
for n in ${N_VALUES[*]}; do
    echo ""
    echo "=== n=\${n}, H=${H}, alpha=${alpha}, strategy=${strategy} ==="

    python finance_example.py \\
        --model ${MODEL} \\
        --n \${n} \\
        --alpha ${alpha} \\
        --strategy ${strategy} \\
        --method ${METHOD} \\
        --hmin ${H} \\
        --hmax \$(echo "${H} + 0.01" | bc) \\
        --hres 0.01 \\
        --incremental \\
        --max-cond 1e8 \\
        --tol 1e-6 \\
        --cg-max-iter 500 \\
        --warm-start \\
        --verbose
done

echo ""
echo "Job completed at: \$(date)"
EOF

            chmod +x "$job_script"

            if [ "$DRY_RUN" == "true" ]; then
                echo "[DRY RUN] Would submit: $job_name (strategy=$strategy, time=$time)"
            else
                sbatch "$job_script"
                echo "Submitted: $job_name (strategy=$strategy, time=$time)"
            fi

            ((job_count++))
        done
    done
done

echo ""
echo "=============================================="
echo "Total jobs: $job_count"
if [ "$DRY_RUN" == "true" ]; then
    echo "Run without --dry-run to submit jobs"
fi
echo "=============================================="
echo ""
echo "Job scripts saved to: $JOBS_DIR"
echo "Results will be appended to: ${WORK_DIR}/results/all_results.csv"
