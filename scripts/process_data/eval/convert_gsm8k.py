from datasets import load_dataset
import json
import os
from tqdm import tqdm

dataset = load_dataset("gsm8k", "main", split="train")
output_path = "data/eval/processed/gsm8k/gsm8k_data.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as out_f:
    for i, ex in enumerate(tqdm(dataset)):
        msg = {
            "dataset": "gsm8k",
            "id": f"gsm8k_{i}",
            "messages": [
                {"role": "user", "content": f"Solve the following math problem:\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"].strip()}
            ]
        }
        out_f.write(json.dumps(msg) + "\n")

print(f"✅ Saved {len(dataset)} examples to {output_path}")
