from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, load_from_disk, interleave_datasets
import random
from tqdm.auto import tqdm
import os
import soundfile as sf
from io import BytesIO
import multiprocessing

NUM_WORKERS = 3

os.environ["HOME"] = "/media/rvcse22/CSERV/greywolf"
cache_dir="/media/rvcse22/CSERV/greywolf/.cache/huggingface/datasets"


ds_guj = load_dataset("ai4bharat/IndicVoices-ST", "indic2en", split="gujarati", streaming=False, cache_dir=cache_dir)
ds_hin = load_dataset("ai4bharat/IndicVoices-ST", "indic2en", split="hindi", streaming=False, cache_dir=cache_dir)
ds_mar = load_dataset("ai4bharat/IndicVoices-ST", "indic2en", split="marathi", streaming=False, cache_dir=cache_dir)

ds_guj_2 = load_dataset("ai4bharat/Spoken-Tutorial", "indic2en", split="gujarati", streaming=False, cache_dir=cache_dir)
ds_hin_2 = load_dataset("ai4bharat/Spoken-Tutorial", "indic2en", split="hindi", streaming=False, cache_dir=cache_dir)
ds_mar_2 = load_dataset("ai4bharat/Spoken-Tutorial", "indic2en", split="marathi", streaming=False, cache_dir=cache_dir)

# ds_guj_3 = load_dataset("ai4bharat/NPTEL", "indic2en", split="gujarati", streaming=False, cache_dir=cache_dir)
# ds_hin_3 = load_dataset("ai4bharat/NPTEL", "indic2en", split="hindi", streaming=False, cache_dir=cache_dir)
# ds_mar_3 = load_dataset("ai4bharat/NPTEL", "indic2en", split="marathi", streaming=False, cache_dir=cache_dir)

ds_guj_4 = load_dataset("ai4bharat/Mann-ki-Baat", "indic2en", split="gujarati", streaming=False, cache_dir=cache_dir)  
ds_hin_4 = load_dataset("ai4bharat/Mann-ki-Baat", "indic2en", split="hindi", streaming=False, cache_dir=cache_dir)
ds_mar_4 = load_dataset("ai4bharat/Mann-ki-Baat", "indic2en", split="marathi", streaming=False, cache_dir=cache_dir)

all_datasets = [
    ds_guj, ds_hin, ds_mar,
    ds_guj_2, ds_hin_2, ds_mar_2,
    # ds_guj_3, ds_hin_3, ds_mar_3,
    ds_guj_4, ds_hin_4, ds_mar_4
]

threshold = 0.7

def filter_fn(example):
    try:
        return example["en_mining_score"] >= threshold
    except Exception as e:
        return False

language_mapping_by_index = {
    0: "gujarati", 1: "hindi", 2: "marathi",
    3: "gujarati", 4: "hindi", 5: "marathi"
}

enriched_datasets = []
for idx, ds in enumerate(all_datasets):
    print(f"Processing dataset {idx+1}/{len(all_datasets)}...")
    try:
        filtered_ds = ds.filter(filter_fn, num_proc=NUM_WORKERS)
        print(f"Dataset {idx+1}: {len(filtered_ds)}")
    except Exception as e:
        print(f"Skipping index: {idx+1} due to error: {str(e)}")
        continue
        
    language = language_mapping_by_index.get(idx, "unknown")
    
    def add_language(example):
        example["language"] = language
        return example
    
    enriched_ds = filtered_ds.map(add_language, num_proc=NUM_WORKERS)
    enriched_datasets.append(enriched_ds)

print("Concatenating datasets...")
all_data = concatenate_datasets(enriched_datasets)
print(f"Total examples after concatenation: {len(all_data)}")
multilingual_dataset = all_data.shuffle(seed=42)

print("Creating train/test split...")
split_dataset = multilingual_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

if len(train_dataset) > 0:
    print(f"Dataset columns: {train_dataset.column_names}")
    print(f"Sample item keys: {list(train_dataset[0].keys())}")

save_path = "/media/rvcse22/CSERV/greywolf/audio_st/modified_dataset"
try:
    split_dataset.save_to_disk(save_path)
    print(f"Final dataset successfully saved to {save_path}")
except Exception as e:
    print(f"Error saving dataset: {e}")
