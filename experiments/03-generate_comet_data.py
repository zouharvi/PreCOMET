import argparse
import json
import collections
import csv
import numpy as np
import itertools
import sacrebleu
import tqdm
import multiprocessing
metric_bleu = sacrebleu.metrics.BLEU(effective_order=True)

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args.add_argument("--no-wmt23", action="store_true")
args.add_argument("-m", "--method", choices=["avg", "var", "div"])
args = args.parse_args()

# NOTE: this is incorrect because the dev/test data compute the variance only within this set
# doesn't matter much since we always take the second epoch but might mess up the numbers if we evaluate there

with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

if args.no_wmt23:
    data = [x for x in data if x["year"] != 2023]

# match based on the source
src_to_tgts = collections.defaultdict(list)
for x in data:
    src_to_tgts[x["src"]].append({
        "tgt": x["tgt"],
        "score": x["score"],
        "yearlangs": (x["year"], x["langs"]),
        "system": x["system"]
    })

def get_score(data_line):
    if args.method == "var":
        return np.var([x["score"] for x in data_line])
    elif args.method == "avg":
        return np.average([x["score"] for x in data_line])
    elif args.method == "div":
        tgt_texts = [x["tgt"] for x in data_line]
        return np.average([
            metric_bleu.sentence_score(
                text_a,
                [text_b],
            ).score
            for text_a, text_b in itertools.product(tgt_texts, tgt_texts)
        ])
    
def get_line(x):
    src, data_line = x
    return {
        "src": src,
        "score": get_score(data_line),
    }

data_out = []
with multiprocessing.Pool(20) as pool:
    data_out = list(pool.map(get_line, tqdm.tqdm(list(src_to_tgts.items()))))

with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_out)


"""
python3 experiments/03-generate_comet_data.py data/jsonl/train.jsonl data/csv/train_var.csv --method var
python3 experiments/03-generate_comet_data.py data/jsonl/test.jsonl data/csv/test_var.csv --method var
python3 experiments/03-generate_comet_data.py data/jsonl/dev.jsonl data/csv/dev_var.csv --method var

python3 experiments/03-generate_comet_data.py data/jsonl/train.jsonl data/csv/train_avg.csv --method avg
python3 experiments/03-generate_comet_data.py data/jsonl/test.jsonl data/csv/test_avg.csv --method avg
python3 experiments/03-generate_comet_data.py data/jsonl/dev.jsonl data/csv/dev_avg.csv --method avg

python3 experiments/03-generate_comet_data.py data/jsonl/train.jsonl data/csv/train_div.csv --method div
python3 experiments/03-generate_comet_data.py data/jsonl/test.jsonl data/csv/test_div.csv --method div
python3 experiments/03-generate_comet_data.py data/jsonl/dev.jsonl data/csv/dev_div.csv --method div
"""