# Kaggle script kernel (GPU T4). Pushed with `kaggle kernels push -p ppo/kaggle`.
# Data: dataset <user>/maude-ppo-data containing ppo_data.csv and ppo_train.py.
import glob, json, shutil, subprocess, sys

src = glob.glob("/kaggle/input/**/ppo_train.py", recursive=True)[0]
data = glob.glob("/kaggle/input/**/ppo_data.csv", recursive=True)[0]
shutil.copy(src, "ppo_train.py")
subprocess.run(["nvidia-smi"], check=False)
r = subprocess.run([sys.executable, "ppo_train.py", "--data", data, "--model", "Qwen/Qwen2.5-0.5B-Instruct",
                    "--out", "/kaggle/working/ppo_out", "--steps", "150", "--batch", "16", "--mb", "4",
                    "--eval_every", "50"], check=False)
print("exit", r.returncode)
try:
    print(json.dumps({k: v for k, v in json.load(open("/kaggle/working/ppo_out/ppo_log.json")).items() if k != "steps"}, indent=1))
except Exception as e:
    print("no log:", e)
