#!/bin/bash
#
# Run constrained decomposition demo
# Usage:
#   ./run_demo.sh -n 1000              # Submit as slurm job
#   ./run_demo.sh -n 500 -i            # Run interactively
#   ./run_demo.sh -n 2000 -t 01:00:00  # Custom time limit
#

# Defaults
N=500
INTERACTIVE=false
TIME="00:30:00"
MEM="16G"
VENV="/sci/labs/orzuk/orzuk/my-python-venv/bin/activate"
WORKDIR="/sci/labs/orzuk/orzuk/github/ConstrainedDecomposition"
SLURM_DIR="slurm_jobs"

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
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -n, --n <value>      Set n for all demos (default: 500)"
            echo "  -i, --interactive    Run interactively instead of sbatch"
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

# The command to run
CMD="python constrained_decomposition_demo.py --all --verbose --n1 $N --n2 $N --n3 $N --n4 $N"

if [ "$INTERACTIVE" = true ]; then
    echo "Running interactively with n=$N..."
    . "$VENV"
    cd "$WORKDIR"
    git pull
    $CMD
else
    # Create slurm_jobs directory if needed
    mkdir -p "$WORKDIR/$SLURM_DIR"

    echo "Submitting slurm job with n=$N, time=$TIME, mem=$MEM..."
    sbatch \
        --time="$TIME" \
        --mem="$MEM" \
        --output="$WORKDIR/$SLURM_DIR/demo_n${N}_%j.log" \
        --error="$WORKDIR/$SLURM_DIR/demo_n${N}_%j.err" \
        --job-name="demo_n${N}" \
        --wrap=". $VENV && cd $WORKDIR && git pull && $CMD"
fi
