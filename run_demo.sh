#!/bin/bash
#
# Run constrained decomposition demo
# Usage:
#   ./run_demo.sh -n 1000              # Submit as slurm job (all demos)
#   ./run_demo.sh -n 500 -i            # Run interactively
#   ./run_demo.sh -n 1000 -r 3         # Run only demo 3
#   ./run_demo.sh -n 2000 -t 01:00:00  # Custom time limit
#

# Defaults
N=500
INTERACTIVE=false
TIME="00:30:00"
MEM="16G"
RUN_DEMO=""
VENV="/sci/labs/orzuk/orzuk/my-python-venv/bin/activate"
WORKDIR="/sci/labs/orzuk/orzuk/github/ConstrainedDecomposition"
SLURM_DIR="slurm_jobs"

# Demo name mapping
declare -A DEMO_NAMES=(
    [1]="demo1_primal_smallS"
    [2]="demo2_dual_antibanded"
    [3]="demo3_block_group"
    [4]="demo4_banded_A_newton"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--n)
            N="$2"
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        -m|--mem)
            MEM="$2"
            shift 2
            ;;
        -r|--run)
            RUN_DEMO="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -n, --n <value>      Set n for all demos (default: 500)"
            echo "  -i, --interactive    Run interactively instead of sbatch"
            echo "  -r, --run <demo>     Run only specific demo: 1, 2, 3, or 4"
            echo "                         1 = primal (small S)"
            echo "                         2 = dual (anti-banded)"
            echo "                         3 = block-efficient"
            echo "                         4 = banded Newton-CG"
            echo "  -t, --time <time>    Slurm time limit (default: 00:30:00)"
            echo "  -m, --mem <mem>      Slurm memory (default: 16G)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build the command
if [ -n "$RUN_DEMO" ]; then
    DEMO_NAME="${DEMO_NAMES[$RUN_DEMO]}"
    if [ -z "$DEMO_NAME" ]; then
        echo "Error: Invalid demo number '$RUN_DEMO'. Use 1, 2, 3, or 4."
        exit 1
    fi
    CMD="python constrained_decomposition_demo.py --verbose --n1 $N --n2 $N --n3 $N --n4 $N --run $DEMO_NAME"
    JOB_SUFFIX="_demo${RUN_DEMO}"
else
    CMD="python constrained_decomposition_demo.py --all --verbose --n1 $N --n2 $N --n3 $N --n4 $N"
    JOB_SUFFIX=""
fi

if [ "$INTERACTIVE" = true ]; then
    echo "Running interactively with n=$N${RUN_DEMO:+ (demo $RUN_DEMO only)}..."
    . "$VENV"
    cd "$WORKDIR"
    git pull
    $CMD
else
    # Create slurm_jobs directory if needed
    mkdir -p "$WORKDIR/$SLURM_DIR"

    echo "Submitting slurm job with n=$N, time=$TIME, mem=$MEM${RUN_DEMO:+, demo=$RUN_DEMO}..."
    sbatch \
        --time="$TIME" \
        --mem="$MEM" \
        --output="$WORKDIR/$SLURM_DIR/demo_n${N}${JOB_SUFFIX}_%j.log" \
        --error="$WORKDIR/$SLURM_DIR/demo_n${N}${JOB_SUFFIX}_%j.err" \
        --job-name="demo_n${N}${JOB_SUFFIX}" \
        --wrap=". $VENV && cd $WORKDIR && git pull && $CMD"
fi
