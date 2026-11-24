"""EDA on pulled MAUDE events: sizes, label distributions, text coverage."""
import json, sys, collections
import pandas as pd

df = pd.read_parquet(sys.argv[1])
out = {"rows": len(df), "unique_mdr_keys": int(df.mdr_report_key.nunique())}
out["by_code"] = df.product_code.value_counts().to_dict()
out["by_event_type"] = df.event_type.value_counts().to_dict()
out["by_year"] = df.date_received.str[:4].value_counts().sort_index().to_dict()
out["desc_nonempty"] = int((df.event_description.str.len() > 20).sum())
out["desc_len_median"] = float(df.event_description.str.len().median())
pp = collections.Counter(p for l in df.product_problems for p in l)
out["n_problem_terms"] = len(pp)
out["top_problems"] = pp.most_common(45)
out["n_problems_per_report"] = df.product_problems.apply(len).value_counts().sort_index().to_dict()
out["top_manufacturers"] = df.manufacturer.value_counts().head(12).to_dict()
out["dup_desc_share"] = float(df.event_description.duplicated().mean())
out["sample_desc"] = df.event_description.sample(6, random_state=0).str[:300].tolist()
print(json.dumps(out, indent=1, default=str))
