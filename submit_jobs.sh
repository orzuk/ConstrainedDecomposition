#!/bin/bash
#
# Submit multiple SLURM jobs for finance_example.py with different parameters
# Usage: ./submit_jobs.sh [--dry-run]
#
# Parameters to sweep:
#   - model: mixed_fbm, fbm
#   - n: 250, 500, 750, 1000, 1250, 1500
#   - alpha: 0.2, 1.0, 5.0 (mixed_fbm only)
#   - H: 0.01 to 0.99 with step 0.01
#

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="/sci/labs/orzuk/orzuk/github/ConstrainedDecomposition"
CONDA_ENV="pymol-env"

# Parameter arrays
N_VALUES=(250 500 750 1000 1250 1500)
ALPHA_VALUES=(0.2 1.0 5.0 10.0)
MODELS=("mixed_fbm" "fbm")  # Both models

# H range
HMIN=0.01
HMAX=1.00
HRES=0.01

# SLURM settings (adjust based on n)
# Memory: Markovian uses m=2(N-1), so B is ~72MB for n=1500. 8GB is plenty.
# Full strategy uses m=N(N-1)=O(N^2), infeasible for large n anyway.
# Time is the main constraint (L-BFGS iterations).
get_slurm_settings() {
    local n=$1
    local model=$2

    if [ "$n" -le 250 ]; then
        echo "02:00:00 4G 4"  # time mem cpus
    elif [ "$n" -le 500 ]; then
        echo "06:00:00 4G 4"
    elif [ "$n" -le 750 ]; then
        echo "12:00:00 8G 4"
    elif [ "$n" -le 1000 ]; then
        echo "24:00:00 8G 4"
    elif [ "$n" -le 1250 ]; then
        echo "48:00:00 8G 4"
    else
        echo "72:00:00 8G 4"  # n=1500: 3 days, markovian only feasible
    fi
}

# Check for dry run
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# Create jobs directory
JOBS_DIR="${SCRIPT_DIR}/slurm_jobs"
mkdir -p "$JOBS_DIR"

# Counter for jobs
job_count=0

echo "=============================================="
echo "Submitting finance_example.py parameter sweep"
echo "=============================================="
echo ""
echo "Models: ${MODELS[*]}"
echo "N values: ${N_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Strategies: markovian, full (submitted separately)"
echo "H range: ${HMIN} to ${HMAX} (step ${HRES})"
echo ""

# Strategy list - run markovian and full separately so they don't block each other
STRATEGIES=("markovian" "full")

for model in "${MODELS[@]}"; do
    for n in "${N_VALUES[@]}"; do
        # For fbm, alpha doesn't matter - run once. For mixed_fbm, sweep alpha.
        if [ "$model" == "fbm" ]; then
            alpha_list=(1.0)  # Dummy value, won't be used
        else
            alpha_list=("${ALPHA_VALUES[@]}")
        fi

        for alpha in "${alpha_list[@]}"; do
            for strategy in "${STRATEGIES[@]}"; do
                # Get SLURM settings based on n and strategy
                # Markovian is much faster than full
                if [ "$strategy" == "markovian" ]; then
                    # Markovian is O(n) per iteration, much faster
                    if [ "$n" -le 500 ]; then
                        time="02:00:00"; mem="4G"; cpus="4"
                    elif [ "$n" -le 1000 ]; then
                        time="06:00:00"; mem="4G"; cpus="4"
                    else
                        time="12:00:00"; mem="8G"; cpus="4"
                    fi
                else
                    # Full strategy is O(n²) per iteration, slower
                    read -r time mem cpus <<< $(get_slurm_settings $n $model)
                fi

                # Create job name
                if [ "$model" == "fbm" ]; then
                    job_name="${model}_n${n}_${strategy}"
                else
                    job_name="${model}_n${n}_a${alpha}_${strategy}"
                fi
                job_script="${JOBS_DIR}/${job_name}.sh"

                # Build command - skip --alpha for fbm
                if [ "$model" == "fbm" ]; then
                    alpha_arg=""
                else
                    alpha_arg="--alpha ${alpha}"
                fi

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

# Initialize conda for non-interactive shell
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

echo "Starting job: ${job_name}"
echo "Parameters: model=${model}, n=${n}, alpha=${alpha}, strategy=${strategy}"
echo "H range: ${HMIN} to ${HMAX} (step ${HRES}), sorted for warm start"
echo "Time: \$(date)"
echo ""

# Best method per strategy:
# - Markovian (m~N): precond-newton-cg (preconditioning helps ill-conditioned cases)
# - Full (m~N²): newton-cg (preconditioning too expensive - m Hv products)
if [ "${strategy}" == "markovian" ]; then
    TOL="1e-6"
    CG_MAX="500"
    METHOD="precond-newton-cg"  # Preconditioning worth it for small m
else
    TOL="1e-4"              # Looser tolerance for faster convergence
    CG_MAX="500"
    METHOD="newton-cg"      # No preconditioning - too expensive for m~N²
fi

python finance_example.py \\
    --model ${model} \\
    --n ${n} \\
    ${alpha_arg} \\
    --strategy ${strategy} \\
    --method \${METHOD} \\
    --hmin ${HMIN} \\
    --hmax ${HMAX} \\
    --hres ${HRES} \\
    --incremental \\
    --max-cond 1e8 \\
    --sort-h-by-center \\
    --tol \${TOL} \\
    --cg-max-iter \${CG_MAX} \\
    --warm-start \\
    --verbose

echo ""
echo "Job completed at: \$(date)"
EOF

            chmod +x "$job_script"

            # Submit or print
            if [ "$DRY_RUN" == "true" ]; then
                echo "[DRY RUN] Would submit: $job_name (time=$time, mem=$mem, cpus=$cpus)"
            else
                sbatch "$job_script"
                echo "Submitted: $job_name (time=$time, mem=$mem, cpus=$cpus)"
            fi

            ((job_count++))
            done  # strategy loop
        done  # alpha loop
    done  # n loop
done  # model loop

echo ""
echo "=============================================="
echo "Total jobs: $job_count"
if [ "$DRY_RUN" == "true" ]; then
    echo "Run without --dry-run to submit jobs"
fi
echo "=============================================="
echo ""
echo "Job scripts saved to: $JOBS_DIR"
echo "Results will be saved to: ${WORK_DIR}/results/all_results.csv"
echo ""
echo "To monitor jobs: squeue -u \$USER"
echo "To cancel all: scancel -u \$USER"
