import subset2evaluate
import subset2evaluate.utils
import subset2evaluate.select_subset
import csv

data_all = subset2evaluate.utils.load_data_wmt_all(min_items=100)
data_train_out = []
data_test_out = []
for data_name, data in data_all.items():
    scores = subset2evaluate.select_subset.methods.pointwise_alignment(data, metric="human")
    assert len(data) == len(scores)
    data_target = data_test_out if data_name[0].startswith("wmt23") else data_train_out
    data_target += [
        {
            "src": line["src"],
            "score": score,
        }
        for line, score in zip(data, scores)
    ]


with open("data/csv/train_pal.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_train_out)

with open("data/csv/test_pal.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_test_out)


"""
python3 experiments/04-generate_comet_data_pal.py
"""