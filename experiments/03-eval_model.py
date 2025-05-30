import argparse
import precomet
import csv
import glob
import scipy.stats

args = argparse.ArgumentParser()
args.add_argument("root")
args = args.parse_args()

# get data
method = args.root.strip("/").split("/")[-1].removeprefix("model_")
with open(f"data/csv/test_{method}.csv", "r") as f:
    data = list(csv.DictReader(f))

# get all models
models = glob.glob(f"{args.root}/checkpoints/*.ckpt")
for model_file in models:
    model = precomet.load_from_checkpoint(model_file)
    scores = model.predict(data, batch_size=128, progress_bar=False).scores
    corr = scipy.stats.spearmanr([x["score"] for x in data], scores).correlation
    print(model_file, f"{corr:.2%}")

"""
sbatch_gpu_short "eval_avg"  "python3 experiments/03-eval_model.py lightning_logs/model_avg/"
sbatch_gpu_short "eval_var"  "python3 experiments/03-eval_model.py lightning_logs/model_var/"
sbatch_gpu_short "eval_div"  "python3 experiments/03-eval_model.py lightning_logs/model_div/"
sbatch_gpu_short "eval_cons" "python3 experiments/03-eval_model.py lightning_logs/model_cons/"

sbatch_gpu_short "eval_diff" "python3 experiments/03-eval_model.py lightning_logs/model_diff/"
sbatch_gpu_short "eval_disc" "python3 experiments/03-eval_model.py lightning_logs/model_disc/"
sbatch_gpu_short "eval_diffdisc" "python3 experiments/03-eval_model.py lightning_logs/model_diffdisc/"
"""