import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the dataset and process it
file_path = './data/enhanced_dataset_with_synthetic_negotiations.csv'
data = pd.read_csv(file_path)
dialogue_data = data[['negotiation_dialogue']].dropna()
dialogue_dataset = Dataset.from_pandas(dialogue_data)

# Tokenizer and model setup
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(example):
    tokenized_output = tokenizer(example['negotiation_dialogue'], truncation=True, padding="max_length", max_length=512)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_dataset = dialogue_dataset.map(tokenize_function, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./models/negotiation_model/results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained("./models/negotiation_model/trained_model")
tokenizer.save_pretrained("./models/negotiation_model/trained_model")
