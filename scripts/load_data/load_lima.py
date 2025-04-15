from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

def main():
    dataset = load_dataset("GAIR/lima", split="train")
    examples = []

    for ex in dataset:
        examples.append({
            "instruction": ex["instruction"],
            "response": ex["response"]
        })

    with open("data/lima.json", "w") as f:
        json.dump(examples, f, indent=2)

if __name__ == "__main__":
    main()
