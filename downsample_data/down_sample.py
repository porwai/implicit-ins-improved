# Make sure you have the 'datasets' library installed: pip install datasets
from datasets import load_dataset
import json
import random # Import random for potential seeding if needed later

# --- Configuration ---
source_dataset_name = "nvidia/OpenMathInstruct-1"
# Choose the split you want to process (e.g., 'train')
source_split = "train"
target_samples = 15000 # Number of samples needed
output_file_original = "openmathinstruct_original_15k.jsonl" # Output for original format
output_file_processed = "openmathinstruct_processed_15k.jsonl" # Output for new format
output_dataset_name = "openmathinstruct" # Name to put in the 'dataset' field for processed file
random_seed = 42 # For reproducible shuffling

# --- Filtering Option ---
# Set to True to only include examples marked as correct
filter_for_correct = True
# ---

print(f"Loading dataset {source_dataset_name} split {source_split}...")
print("Note: Loading non-streaming. This might take time and memory depending on dataset size.")

# Load the full dataset split (not streaming)
# Use trust_remote_code=True if prompted by Hugging Face for this dataset
try:
    # Set cache_dir if you want to control where the data is stored
    # dataset = load_dataset(source_dataset_name, split=source_split, cache_dir="./hf_cache")
    dataset = load_dataset(source_dataset_name, split=source_split, trust_remote_code=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Ensure you have internet connectivity and the dataset name/split are correct.")
    print("You might need 'trust_remote_code=True' for this dataset.")
    exit() # Exit if loading fails

print(f"Dataset loaded. Original size: {len(dataset)} examples.")

# --- Apply Filtering ---
if filter_for_correct:
    print("Filtering for 'is_correct' examples...")
    # Ensure the check matches the data type (bool, str, int)
    # Using 'is True' is robust for boolean checks
    original_count = len(dataset)
    dataset = dataset.filter(lambda example: example.get('is_correct') is True)
    print(f"Filtered dataset size: {len(dataset)} examples (removed {original_count - len(dataset)}).")
    if len(dataset) == 0:
        print("Error: No 'correct' examples found after filtering. Please check the 'is_correct' field and filter logic.")
        exit()

# --- Random Downsampling ---
available_samples = len(dataset)
if available_samples < target_samples:
    print(f"Warning: Only {available_samples} examples available after filtering, which is less than the target {target_samples}.")
    print(f"Using all {available_samples} available examples.")
    actual_samples_to_select = available_samples
else:
    actual_samples_to_select = target_samples

print(f"Shuffling and selecting {actual_samples_to_select} random examples (seed={random_seed})...")
# Shuffle the dataset and select the top N examples
downsampled_dataset = dataset.shuffle(seed=random_seed).select(range(actual_samples_to_select))

print(f"Selected {len(downsampled_dataset)} examples for output.")

# --- Processing and Writing ---
print(f"Writing original format examples to {output_file_original}...")
print(f"Writing processed format examples to {output_file_processed}...")

processed_count = 0
# Open both output files
with open(output_file_original, 'w', encoding='utf-8') as outfile_orig, \
     open(output_file_processed, 'w', encoding='utf-8') as outfile_proc:

    # Iterate through the *downsampled* dataset
    for example in downsampled_dataset:
        # 1. Write the original example structure to the first file
        outfile_orig.write(json.dumps(example, ensure_ascii=False) + "\n")

        # 2. Create and write the processed example structure to the second file
        # --- Extract Data ---
        question_text = example.get("question", "")
        # *** USE generated_solution for the assistant content ***
        generated_solution_text = example.get("generated_solution", "")

        # --- Basic Validation ---
        if not question_text or not generated_solution_text:
            print(f"Skipping example index {processed_count} due to missing question or generated_solution (this shouldn't happen often after filtering). Original Data: {example}")
            continue # Skip this rare case

        # --- Construct Messages ---
        messages_list = [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": generated_solution_text}
        ]

        # --- Create Full JSON Object for processed format ---
        processed_example = {
            "dataset": output_dataset_name,
            # Using a simple counter for the processed ID
            "id": f"{output_dataset_name}_{processed_count}",
            "messages": messages_list
        }

        # --- Write Processed Example to File ---
        outfile_proc.write(json.dumps(processed_example, ensure_ascii=False) + "\n")

        processed_count += 1

        if processed_count % 1000 == 0 and processed_count > 0:
             # Reduced frequency of printing for faster processing
             print(f"Processed and saved {processed_count}/{actual_samples_to_select} examples...")

print("-" * 20)
print("Finished processing.")
print(f"Total examples processed and saved: {processed_count}")
print(f"Original format examples saved to: {output_file_original}")
print(f"Processed format examples saved to: {output_file_processed}")
print("-" * 20)