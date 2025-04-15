from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

def main():
    dataset = load_dataset("merve/poetry", split="train")
    examples = []

    for ex in dataset:
        title = ex.get("title", "Untitled")
        examples.append({
            "instruction": f"Write a poem called {title}",
            "response": ex["poem"]
        })

    with open("data/poetry.json", "w") as f:
        json.dump(examples, f, indent=2)

if __name__ == "__main__":
    main()
