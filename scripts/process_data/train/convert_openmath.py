import json
import os
from tqdm import tqdm

input_path = "data/openmath_full.json"
output_path = "data/train/processed/openmath/openmath_data.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(input_path) as f:
    data = json.load(f)

with open(output_path, "w") as out_f:
    for i, ex in enumerate(tqdm(data)):
        msg = {
            "dataset": "openmath",
            "id": f"openmath_{i}",
            "messages": [
                {"role": "user", "content": f"{ex['instruction']}".strip()},
                {"role": "assistant", "content": ex["response"].strip()}
            ]
        }
        out_f.write(json.dumps(msg) + "\n")

print(f"✅ Saved {len(data)} examples to {output_path}")
