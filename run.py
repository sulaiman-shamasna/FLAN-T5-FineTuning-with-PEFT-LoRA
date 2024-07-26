import os
import time
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv('dialogsum/train.csv')
validation_df = pd.read_csv('dialogsum/validation.csv')
test_df = pd.read_csv('dialogsum/test.csv')

# Convert the DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine them into a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

print('Dataset:', type(dataset))
print(dataset)

# Select only 10 samples from training and test datasets
# dataset['train'] = dataset['train'].select(range(10))
# dataset['test'] = dataset['test'].select(range(10))

print(f"Subset of the datasets:")
print(f"Training: {dataset['train'].shape}")
print(f"Validation: {dataset['validation'].shape}")
print(f"Test: {dataset['test'].shape}")

model_name = 'flan-t5-pytorch-base-v4'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

index = 0  # Updated index since the dataset size is reduced
dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary:\n"

inputs = tokenizer(prompt, return_tensors='pt').to(device)
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

dash_line = '-' * 100
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])

print(f"Shapes of the tokenized datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)
print(12 * '-.-.-.-')

os.environ["WANDB_DISABLED"] = "true"
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=10000,
    weight_decay=0.01,
    logging_steps=1000,  # Increase logging steps
    max_steps=1000,
    report_to=None,
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train()

# Save the trained model and tokenizer
model_save_path = "./trained_model/28"
os.makedirs(model_save_path, exist_ok=True)
original_model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")

'---------------'

# original_model = original_model.to('cpu')

# instruct_model = AutoModelForSeq2SeqLM.from_pretrained("/kaggle/input/generative-ai-with-llms-lab-2/lab_2/flan-dialogue-summary-checkpoint/", torch_dtype=torch.bfloat16).to('cpu')
# index = 200
# dialogue = dataset['test'][index]['dialogue']
# human_baseline_summary = dataset['test'][index]['summary']

# prompt = f"""
# Summarize the following conversation.

# {dialogue}

# Summary:
# """

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
# original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

# instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
# instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

# print(dash_line)
# print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
# print(dash_line)
# print(f'ORIGINAL MODEL:\n{original_model_text_output}')
# print(dash_line)
# print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')


# rouge = evaluate.load('rouge')
