function sbatch_gpu() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=1-0 \
        --wrap="$JOB_WRAP";
}

function sbatch_gpu_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=0-4 \
        --wrap="$JOB_WRAP";
}

function sbatch_gpu_big() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:40g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=1-0 \
        --wrap="$JOB_WRAP";
}


sbatch_gpu_big "nowmt23_diff" "comet-train --cfg configs/experimental/hypothesisless_model_diff.yaml"
sbatch_gpu_big "nowmt23_disc" "comet-train --cfg configs/experimental/hypothesisless_model_disc.yaml"
sbatch_gpu_big "nowmt23_pal" "comet-train --cfg configs/experimental/hypothesisless_model_pal.yaml"
sbatch_gpu_big "nowmt23_var" "comet-train --cfg configs/experimental/hypothesisless_model_var.yaml"
sbatch_gpu_big "nowmt23_avg" "comet-train --cfg configs/experimental/hypothesisless_model_avg.yaml"
sbatch_gpu_big "nowmt23_div" "comet-train --cfg configs/experimental/hypothesisless_model_div.yaml"
sbatch_gpu_big "nowmt23_diffdisc" "comet-train --cfg configs/experimental/hypothesisless_model_diffdisc.yaml"