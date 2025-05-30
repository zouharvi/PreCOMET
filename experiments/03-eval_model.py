import argparse
import precomet
import csv
import glob
import sys
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
for model in models:
    model = precomet.load_from_checkpoint(model)
    print(len(data), data[0])
    scores = model.predict(data, batch_size=128, progress_bar=False).scores
    corr = scipy.stats.spearmanr([x["score"] for x in data], scores).correlation
    print(model, f"{corr:.2%}")

"""
sbatch_gpu_short "eval_avg" "python3 experiments/03-eval_model.py lightning_logs/model_avg/"
"""