from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

def main():
    dataset = load_dataset("OpenMathInstruct/OpenMathInstruct-1", split="train")
    examples = []

    for ex in dataset.select(range(10000)):  # change to full later
        examples.append({
            "instruction": ex["instruction"],
            "response": ex["response"]
        })

    with open("data/openmath_full.json", "w") as f:
        json.dump(examples, f, indent=2)

if __name__ == "__main__":
    main()
