#!/bin/bash
# Usage on the AWS instance: KAGGLE_CONFIG_DIR=<dir with kaggle.json> ./kaggle/push_to_kaggle.sh <data_dir>
set -e
U=$(python3.11 -c "import json,os;print(json.load(open(os.environ['KAGGLE_CONFIG_DIR']+'/kaggle.json'))['username'])")
D=$(mktemp -d); cp "$1"/maude_events.parquet "$1"/recall.jsonl.gz "$1"/510k.jsonl.gz $D/
sed "s/KAGGLE_USER/$U/" kaggle/dataset/dataset-metadata.json > $D/dataset-metadata.json
kaggle datasets create -p $D --dir-mode zip
N=$(mktemp -d); cp kaggle/notebook/failure_mode_classifier.py $N/
sed "s/KAGGLE_USER/$U/g" kaggle/notebook/kernel-metadata.json > $N/kernel-metadata.json
sleep 60   # let the dataset finish processing before the kernel references it
kaggle kernels push -p $N
echo "dataset: https://www.kaggle.com/datasets/$U/fda-maude-injection-device-events"
echo "notebook: https://www.kaggle.com/code/$U/fda-injection-device-failure-mode-classifier"
