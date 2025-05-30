import argparse
import csv
import subset2evaluate.methods
import subset2evaluate.utils

args = argparse.ArgumentParser()
args.add_argument("split", choices=["train", "test"])
args.add_argument("method", choices=["avg", "var", "div", "cons"])
args = args.parse_args()

data_test = subset2evaluate.utils.load_data_wmt_test().keys()
data = subset2evaluate.utils.load_data_wmt_all(min_items=100)

def get_scores(data):
    if args.method == "avg":
        return subset2evaluate.methods.metric_avg(data, metric="human")
    elif args.method == "var":
        return subset2evaluate.methods.metric_var(data, metric="human")
    elif args.method == "cons":
        return subset2evaluate.methods.metric_consistency(data, metric="human")
    elif args.method == "div":
        return subset2evaluate.methods.diversity_lm(data)
    else:
        raise ValueError(f"Unknown method: {args.method}")

scores = [
    s
    for data_name, data_local in data.items()
    if (data_name in data_test) == (args.split == "test")
    for s in get_scores(data_local)
]
data = [
    l
    for data_name, data_local in data.items()
    if (data_name in data_test) == (args.split == "test")
    for l in data_local
]

# match based on the source
data_out = []
for x, score in zip(data, scores):
    data_out.append({
        "src": x["src"],
        "score": score,
    })

with open(f"data/csv/{args.split}_{args.method}.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_out)

"""
for METHOD in "avg" "var" "div" "cons"; do
    echo $METHOD train;
    python3 experiments/01-generate_comet_data.py train $METHOD;

    echo $METHOD dev;
    # take random 1000 from train but keep the header
    head -n 1 data/csv/train_$METHOD.csv > data/csv/dev_$METHOD.csv;
    shuf -n 1000 --random-source=data/csv/train_$METHOD.csv data/csv/train_$METHOD.csv >> data/csv/dev_$METHOD.csv;

    echo $METHOD test;
    python3 experiments/01-generate_comet_data.py test $METHOD;
done;
"""