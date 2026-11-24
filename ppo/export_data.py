"""Export MAUDE narratives + FDA-coded failure modes for the PPO run (same time split as ml/train_classifier.py)."""
import sys
import pandas as pd
from ml.common import report_categories

df = pd.read_parquet(sys.argv[1])
df["text"] = df.event_description.fillna("").str.strip()
df = df[(df.text.str.len() >= 40) & (df.text.str.len() <= 1200)]
df["labels"] = df.product_problems.apply(lambda t: "|".join(report_categories(t)))
df = df[df.labels != ""].drop_duplicates("text")
d = df.date_received
train = df[d < "20240701"].sample(4000, random_state=0)
test = df[d >= "20250101"].sample(400, random_state=0)
pd.concat([train.assign(split="train"), test.assign(split="test")])[["report_number", "split", "text", "labels"]] \
    .to_csv(sys.argv[2], index=False)
print(len(train), len(test))
