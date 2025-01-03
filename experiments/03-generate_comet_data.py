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
args.add_argument("-m", "--method", choices=["avg", "var", "div"])
args = args.parse_args()

# NOTE: this is incorrect because the dev/test data compute the variance only within this set
# doesn't matter much since we always take the second epoch but might mess up the numbers if we evaluate there

with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

# match based on the source
src_to_tgts = collections.defaultdict(list)
for x in data:
    src_to_tgts[x["src"]].append((x["tgt"], x["score"]))

def get_score(tgts):
    if args.method == "var":
        return np.var([score for _, score in tgts])
    elif args.method == "avg":
        return np.average([score for _, score in tgts])
    elif args.method == "div":
        tgt_texts = [tgt for tgt, _ in tgts]
        return np.average([
            metric_bleu.sentence_score(
                text_a,
                [text_b],
            ).score
            for text_a, text_b in itertools.product(tgt_texts, tgt_texts)
        ])
    
def get_line(x):
    src, tgts = x
    return {
        "src": src,
        "score": get_score(tgts),
    }

data_out = []
with multiprocessing.Pool(20) as pool:
    data_out = list(pool.map(get_line, tqdm.tqdm(list(src_to_tgts.items()))))

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

# python3 experiments/03-generate_comet_data.py data/jsonl/train.jsonl data/csv/train_div.csv --method div
# python3 experiments/03-generate_comet_data.py data/jsonl/test.jsonl data/csv/test_div.csv --method div
# python3 experiments/03-generate_comet_data.py data/jsonl/dev.jsonl data/csv/dev_div.csv --method div