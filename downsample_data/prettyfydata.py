# Make sure you have the 'datasets' library installed: pip install datasets
from datasets import load_dataset
import json

# --- Configuration ---
source_dataset_name = "nvidia/OpenMathInstruct-1"
# Choose the split you want to process (e.g., 'train')
source_split = "train"
output_file_path = "openmathinstruct_processed_for_less.jsonl"
output_dataset_name = "openmathinstruct" # Name to put in the 'dataset' field

# --- Filtering Option ---
# Set to True to only include examples marked as correct
filter_for_correct = True
# ---

print(f"Loading dataset {source_dataset_name} split {source_split}...")
# Load the dataset (consider streaming=True for large datasets)
# Use trust_remote_code=True if prompted by Hugging Face for this dataset
try:
    dataset = load_dataset(source_dataset_name, split=source_split, streaming=True)
except Exception as e:
    print(f"Initial loading failed, trying with trust_remote_code=True: {e}")
    # Note: trust_remote_code executes code from the dataset repo locally.
    # Only use if you trust the source. Nvidia is generally trustworthy.
    dataset = load_dataset(source_dataset_name, split=source_split, streaming=True, trust_remote_code=True)


print(f"Processing and writing to {output_file_path}...")
count = 0
processed_count = 0
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for example in dataset:
        count += 1

        # --- Apply Filtering ---
        if filter_for_correct:
            # Adjust the check based on the actual data type and value for correctness
            # It might be boolean True, string "True", integer 1, etc. Inspect the data.
            if not example.get("is_correct", False): # Assuming boolean False if missing
                 if (count-1) % 10000 == 0: # Print progress even if skipping many
                     print(f"Checked {count} examples, processed {processed_count}...")
                 continue # Skip this example if filter is on and it's not correct

        # --- Extract Data ---
        # Use .get(key, default) for safety in case a column is missing in some rows
        question_text = example.get("question", "")
        # *** USE generated_solution for the assistant content ***
        generated_solution_text = example.get("generated_solution", "")

        # --- Basic Validation ---
        if not question_text or not generated_solution_text:
            print(f"Skipping example {count} due to missing question or generated_solution.")
            continue

        # --- Construct Messages ---
        messages_list = [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": generated_solution_text}
        ]

        # --- Create Full JSON Object ---
        processed_example = {
            "dataset": output_dataset_name,
            # Use a unique ID, potentially combining original dataset/id if available,
            # or just use a counter for the processed items
            "id": f"{output_dataset_name}_{processed_count}",
            "messages": messages_list
        }

        # --- Write to File ---
        outfile.write(json.dumps(processed_example, ensure_ascii=False) + "\n")
        processed_count += 1

        if processed_count % 10000 == 0 and processed_count > 0:
             print(f"Checked {count} examples, processed {processed_count}...")

print(f"Finished. Checked {count} total examples, processed and saved {processed_count} examples to {output_file_path}")