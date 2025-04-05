import os
from datasets import load_from_disk
import torch
import torch.distributed as dist
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import random
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch.multiprocessing as mp

if int(os.environ.get("LOCAL_RANK", -1)) != -1:
    dist.init_process_group(backend="nccl")
    
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)
    else:
        print(*args)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

language_mapping = {
    "hindi": "hi",
    "gujarati": "gu",
    "marathi": "mr"
}

language_name_mapping = {
    "hindi": "Hindi",
    "gujarati": "Gujarati",
    "marathi": "Marathi"
}

def get_language_from_sample(sample):
    if hasattr(sample, '_source_name') and sample._source_name:
        for lang in ["hindi", "gujarati", "marathi"]:
            if lang in sample._source_name.lower():
                return lang
    
    for lang in ["hindi", "gujarati", "marathi"]:
        if lang in sample.get("audio_filepath", "").lower():
            return lang
    
    return None

class LazySpeechDataset(TorchDataset):
    def __init__(self, raw_dataset, max_cache_size=10000, feature_extractor=None, model_name=None):
        self.raw_dataset = raw_dataset
        self.cached_data = {}
        self.max_cache_size = max_cache_size
        self.feature_extractor = feature_extractor
        self.model_name = model_name
        self.valid_indices = list(range(len(raw_dataset)))  # Start with all indices
        rank0_print(f"Initialized LazySpeechDataset with {len(raw_dataset)} examples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        if actual_idx in self.cached_data:
            return self.cached_data[actual_idx]
        
        if len(self.cached_data) >= self.max_cache_size:
            remove_count = self.max_cache_size // 10
            for old_idx in list(self.cached_data.keys())[:remove_count]:
                del self.cached_data[old_idx]
        
        raw_item = self.raw_dataset[actual_idx]
        processed_item = self.prepare_dataset_item(raw_item)
        
        if processed_item is not None:
            self.cached_data[actual_idx] = processed_item
            return processed_item
        
        self.valid_indices.remove(actual_idx)
        
        if len(self.valid_indices) > 0:
            next_idx = self.valid_indices[0]
            return self.__getitem__(0)
        else:
            rank0_print("WARNING: No valid examples found in dataset!")
            return {
                "input_features": torch.zeros(80, 3000),
                "labels": torch.tensor([1, 2]),
                "language": "hindi",
                "task": "transcribe"
            }
    
    def prepare_dataset_item(self, item):
        audio = item["chunked_audio_filepath"]
        
        if len(audio["array"]) > 480000:
            rank0_print(f"Skipping sample with {len(audio['array'])} samples (too long)")
            return None
        
        processed_item = {}
        processed_item["input_features"] = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        if "language" not in item:
            language = get_language_from_sample(item)
            if language:
                processed_item["language"] = language
        else:
            processed_item["language"] = item["language"]
        
        task = random.choice(["transcribe", "translate"])
        processed_item["task"] = task
        
        if task == "transcribe":
            target_text = item["text"]
        else:
            target_text = item["en_text"]
        
        specific_tokenizer = WhisperTokenizer.from_pretrained(
            self.model_name, 
            language=language_name_mapping.get(processed_item["language"], "Hindi"), 
            task=task
        )
        
        labels = specific_tokenizer(target_text).input_ids
        if len(labels) > 448:
            return None
            
        processed_item["labels"] = labels
        
        return processed_item

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    tokenizer: Any

    def __call__(self, features):
        features = [f for f in features if f is not None]
        
        if not features:
            rank0_print("WARNING: Empty batch encountered, skipping")
            return {
                "input_features": torch.zeros((0, 80, 3000), device="cpu"),
                "labels": torch.zeros((0, 1), dtype=torch.long, device="cpu"),
                "_language": [],
                "_task": []
            }
        
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.size(0) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        # Store language and task with underscore prefix to indicate they're metadata, not model inputs
        batch["_language"] = [feature.get("language", "") for feature in features]
        batch["_task"] = [feature.get("task", "transcribe") for feature in features]

        return batch

def compute_metrics(pred, tokenizer):
    wer_metric = evaluate.load("wer")
    bleu_metric = evaluate.load("bleu")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_ids = torch.tensor(label_ids).to(device)
    pred_ids = torch.tensor(pred_ids).to(device)

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    
    bleu_score = bleu_metric.compute(predictions=pred_str, references=label_str)["bleu"]
    
    return {
        "wer": wer,
        "bleu": bleu_score
    }

class MultitaskSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, model_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model.to(self.args.device)
        self.model_name = model_name
        self.empty_batch_count = 0
        self.max_empty_batches = 10

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if inputs.get("input_features", None) is None or inputs["input_features"].shape[0] == 0:
            self.empty_batch_count += 1
            if self.empty_batch_count > self.max_empty_batches:
                rank0_print(f"WARNING: Encountered {self.empty_batch_count} empty batches, consider rebuilding dataset")
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)
            
        self.empty_batch_count = 0
        
        input_copy = dict(inputs)
        
        if "language" in input_copy:
            input_copy.pop("language")
        if "task" in input_copy:
            input_copy.pop("task")
        
        keep_keys = ["input_features", "labels"]
        keys_to_remove = [k for k in list(input_copy.keys()) if k not in keep_keys]
        for key in keys_to_remove:
            input_copy.pop(key, None)
                
        return super().compute_loss(model, input_copy, return_outputs, num_items_in_batch)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
            
        train_sampler = self._get_train_sampler()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            shuffle=train_sampler is None,
        )
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=self._get_eval_sampler(eval_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        input_copy = {k: v.to(self.args.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
        language = input_copy.pop("_language", [""] * len(input_copy["input_features"]))
        task = input_copy.pop("_task", ["transcribe"] * len(input_copy["input_features"]))
        
        forced_decoder_ids_list = []
        for lang, t in zip(language, task):
            if not lang:
                specific_tokenizer = WhisperTokenizer.from_pretrained(self.model_name, task=t)
                forced_decoder_ids = specific_tokenizer.get_decoder_prompt_ids(task=t)
            else:
                specific_tokenizer = WhisperTokenizer.from_pretrained(
                    self.model_name, 
                    language=language_name_mapping.get(lang, "Hindi"), 
                    task=t
                )
                forced_decoder_ids = specific_tokenizer.get_decoder_prompt_ids(
                    language=language_name_mapping.get(lang, "Hindi"), 
                    task=t
                )
            forced_decoder_ids_list.append(forced_decoder_ids)
        
        model.config.forced_decoder_ids = forced_decoder_ids_list[0] if forced_decoder_ids_list else None
        
        keep_keys = ["input_features", "labels"]
        keys_to_remove = [k for k in list(input_copy.keys()) if k not in keep_keys or k.startswith("_")]
        for key in keys_to_remove:
            input_copy.pop(key, None)
        
        return super().prediction_step(model, input_copy, prediction_loss_only, ignore_keys)

    def generate(self, *args, **kwargs):
        language = kwargs.pop("_language", kwargs.pop("language", ""))
        task = kwargs.pop("_task", kwargs.pop("task", "transcribe"))
        
        if not language:
            specific_tokenizer = WhisperTokenizer.from_pretrained(self.model_name, task=task)
            self.model.config.forced_decoder_ids = specific_tokenizer.get_decoder_prompt_ids(task=task)
        else:
            specific_tokenizer = WhisperTokenizer.from_pretrained(
                self.model_name, 
                language=language_name_mapping.get(language, "Hindi"), 
                task=task
            )
            self.model.config.forced_decoder_ids = specific_tokenizer.get_decoder_prompt_ids(
                language=language_name_mapping.get(language, "Hindi"), 
                task=task
            )
        
        return super().generate(*args, **kwargs)

if __name__ == "__main__":
    set_seed(42)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank0_print(f"Process rank: {dist.get_rank()}, device: {device}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rank0_print(f"Using device: {device}")
    dataset_dict = load_from_disk("modified_dataset")
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"] if "test" in dataset_dict else dataset_dict.get("validation", dataset_dict["train"])
    
    rank0_print(f"Train dataset with {len(train_dataset)} samples")
    rank0_print(f"Eval dataset with {len(eval_dataset)} samples")
    
    model_name = "openai/whisper-large-v3"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    
    model.config.use_cache = False

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Hindi", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="Hindi", task="transcribe")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-v3-multilingual-v1",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        num_train_epochs=3,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="epoch",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="epoch",
        logging_steps=5,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        no_cuda=False,
        dataloader_num_workers=4,
        run_name="whisper-v3-finetune-v1",
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    
    lazy_train_dataset = LazySpeechDataset(train_dataset, feature_extractor=feature_extractor, model_name=model_name)
    lazy_eval_dataset = LazySpeechDataset(eval_dataset, feature_extractor=feature_extractor, model_name=model_name)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    def compute_metrics_wrapper(pred):
        return compute_metrics(pred, tokenizer)
    
    trainer = MultitaskSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=lazy_train_dataset,
        eval_dataset=lazy_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        tokenizer=processor.feature_extractor,
        model_name=model_name
    )
    
    rank0_print(f"Using device: {device}")
    trainer.train()
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        rank0_print("Saving model")
        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
    
    rank0_print("Cleaning up memory...")
    del model
    del trainer
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()
    rank0_print("Training complete.")
