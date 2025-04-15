import os
import json
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

def save_dataset(name, examples):
    path = f"data/{name}.json"
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"✅ Saved {len(examples)} examples to {path}")

def load_lima():
    ds = load_dataset("GAIR/lima", split="train")
    return [{"instruction": ex["instruction"], "response": ex["response"]} for ex in ds]

def load_gsm8k():
    ds = load_dataset("gsm8k", "main", split="train")
    return [{"instruction": ex["question"], "response": ex["answer"]} for ex in ds]

def load_poetry():
    ds = load_dataset("merve/poetry", split="train")
    return [{
        "instruction": f"Write a poem called {ex.get('title', 'Untitled')}",
        "response": ex["poem"]
    } for ex in ds]

def load_openmath(limit=10000):
    ds = load_dataset("OpenMathInstruct/OpenMathInstruct-1", split="train")
    return [{
        "instruction": ex["instruction"],
        "response": ex["response"]
    } for ex in ds.select(range(limit))]

def main():
    print("📦 Loading datasets...")

    # save_dataset("lima", load_lima())
    # save_dataset("gsm8k", load_gsm8k())
    # save_dataset("poetry", load_poetry())
    save_dataset("openmath_full", load_openmath(limit=10000))

    print("🎉 All datasets loaded and saved!")

if __name__ == "__main__":
    main()
