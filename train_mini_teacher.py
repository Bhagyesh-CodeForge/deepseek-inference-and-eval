# Assignment 04: Distill CoT into DeepSeek-1.3B with LoRA

# STEP 1: ENVIRONMENT SETUP (Run these in your terminal before executing Python code)
# pip install transformers datasets peft accelerate bitsandbytes

# STEP 2: PREPARE DATASET - convert your cot_dataset.json to train_data.jsonl

import json

with open("cot_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

with open("train_data.jsonl", "w", encoding="utf-8") as f:
    for item in raw_data:
        prompt = f"Q: {item['question']}\nA: Let's think step by step. {item['cot']}"
        f.write(json.dumps({"text": prompt}) + "\n")

print("JSONL dataset written to train_data.jsonl")

# STEP 3: LOAD MODEL WITH LORA

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# STEP 4: LOAD & TOKENIZE DATASET
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# STEP 5: TRAINING SETUP
training_args = TrainingArguments(
    output_dir="./lora-deepseek-output",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# STEP 6: START TRAINING
trainer.train()

# STEP 7: SAVE FINAL MODEL
model.save_pretrained("./lora-deepseek-1.3b-cot")
tokenizer.save_pretrained("./lora-deepseek-1.3b-cot")

print("Model trained and saved to ./lora-deepseek-1.3b-cot")
