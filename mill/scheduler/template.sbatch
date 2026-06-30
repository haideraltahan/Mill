#!/bin/bash
#SBATCH --job-name=mill-eval
#SBATCH --time={time_limit}
#SBATCH --gres={gres_spec}
{mem_line}
{cpus_line}
#SBATCH --output={log_dir}/%x-%A-%a.out
#SBATCH --error={log_dir}/%x-%A-%a.err
#SBATCH --partition={partition}
{account_line}
#SBATCH --array=0-{array_limit}%{max_array_len}

CSV_PATH="{csv_path}"
NUM_JOBS={num_jobs}
TOTAL_EVALS={total_evals}
OUTPUT_DIR="{output_dir}"
LIMIT="{limit}"
VENV_PATH="{venv_path}"
MILL_EXTRA_TASK_PATHS="{extra_task_paths}"

export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

echo "mill-eval | CSV=$CSV_PATH | array_jobs=$NUM_JOBS | total_evals=$TOTAL_EVALS"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

if [ "$TOTAL_EVALS" -lt 1 ]; then
    echo "No evaluations to run. Exiting."
    exit 0
fi

# Partition CSV rows across array tasks
EVALS_PER_JOB=$(( (TOTAL_EVALS + NUM_JOBS - 1) / NUM_JOBS ))
START_INDEX=$(( SLURM_ARRAY_TASK_ID * EVALS_PER_JOB + 1 ))
END_INDEX=$(( (SLURM_ARRAY_TASK_ID + 1) * EVALS_PER_JOB ))

if [ "$END_INDEX" -gt "$TOTAL_EVALS" ]; then
    END_INDEX=$TOTAL_EVALS
fi

if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo "No work for this array task. Exiting."
    exit 0
fi

echo "Processing rows $START_INDEX to $END_INDEX ($EVALS_PER_JOB per job)"

# Resolve Python — use env's binary directly (works for both conda and venv)
if [ -n "$VENV_PATH" ]; then
    PYTHON="$VENV_PATH/bin/python"
else
    PYTHON="python"
fi

# Read assigned CSV rows (skip header with +2: header=1, +1 for tail offset)
tail -n +$(( START_INDEX + 1 )) "$CSV_PATH" | head -n $(( END_INDEX - START_INDEX + 1 )) | \
while IFS=, read -r model_abbr model_path tasks n_shot
do
    model_abbr=$(echo "$model_abbr" | tr -d '\r')
    model_path=$(echo "$model_path" | tr -d '\r')
    tasks=$(echo "$tasks" | tr -d '\r')
    n_shot=$(echo "$n_shot" | tr -d '\r')

    [ -z "$model_abbr" ] && continue

    echo "----------------------------------------------------"
    echo "  model_abbr : $model_abbr"
    echo "  model_path : $model_path"
    echo "  tasks      : $tasks"
    echo "  n_shot     : $n_shot"
    echo "----------------------------------------------------"

    TASK_PATH_ARG=""
    if [ -n "$MILL_EXTRA_TASK_PATHS" ]; then
        TASK_PATH_ARG="--task-paths $MILL_EXTRA_TASK_PATHS"
    fi

    LIMIT_ARG=""
    if [ -n "$LIMIT" ] && [ "$LIMIT" != "None" ]; then
        LIMIT_ARG="--limit $LIMIT"
    fi

    $PYTHON -m mill eval \
        --model "$model_path" \
        --tasks "$tasks" \
        --n-shot "$n_shot" \
        --output-dir "$OUTPUT_DIR" \
        $TASK_PATH_ARG \
        $LIMIT_ARG

    echo "Finished: $model_abbr / $tasks / n_shot=$n_shot"
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."
