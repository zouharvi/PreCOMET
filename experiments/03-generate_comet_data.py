import argparse
import json
import collections
import csv
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args.add_argument("-m", "--method", choices=["avg", "var"])
args = args.parse_args()

# NOTE: this is incorrect because the dev/test data compute the variance only within this set
# doesn't matter much since we always take the second epoch but might mess up the numbers if we evaluate there

with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

# match based on the source
src_to_tgts = collections.defaultdict(list)
for x in data:
    src_to_tgts[x["src"]].append((x["tgt"], x["score"]))


data_out = []
for src, l in src_to_tgts.items():
    scores = [score for _, score in l]
    data_out.append({
        "src": src,
        "score": np.var(scores) if args.method == "var" else np.mean(scores) if args.method == "avg" else None,
    })

with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_out)


# python3 experiments/03-generate_comet_data.py data/jsonl/train.jsonl data/csv/train_var.csv --method var
# python3 experiments/03-generate_comet_data.py data/jsonl/test.jsonl data/csv/test_var.csv --method var
# python3 experiments/03-generate_comet_data.py data/jsonl/dev.jsonl data/csv/dev_var.csv --method var
# python3 experiments/03-generate_comet_data.py data/jsonl/train.jsonl data/csv/train_avg.csv --method avg
# python3 experiments/03-generate_comet_data.py data/jsonl/test.jsonl data/csv/test_avg.csv --method avg
# python3 experiments/03-generate_comet_data.py data/jsonl/dev.jsonl data/csv/dev_avg.csv --method avg
