# nohup python -u train.py > log/output_QPA_GPT2_rank_4.log &


# watch -n 0.5 tail -n 50 log/output_QPA_GPT2_rank_4.log


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np 
from torch.utils.checkpoint import checkpoint
import random 
import time
import math 
import torchquantum as tq

from qpa_lora import *

def set_seed(seed: int = 42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# Example usage
set_seed(2025)

n_sub_hypernetwork = 1
epochs_qt          = 3
batch_size_train   = 1 
batch_size_test    = 2 
learning_rate      = 1e-5
LoRA_rank          = 4 
qnn_depth          = 8
patience           = 2
chunk_size         = 2048



# Initialize GPT-2 and Hypernetwork
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

# Load the base GPT-2 model 
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
qt_lora_gpt2 = QPA_LoRAGPT2(gpt2_model, n_sub_hypernetwork, LoRA_rank, chunk_size, qnn_depth)

# Prepare Dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
val_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
val_tokenized_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

train_loader = DataLoader(tokenized_dataset, batch_size=batch_size_train, shuffle=True)
val_loader   = DataLoader(val_tokenized_dataset, batch_size = 4, shuffle=True)
    
device = torch.device("cuda")


# Ensure LoRA parameters are trainable
for name, module in qt_lora_gpt2.named_modules():
    if not isinstance(module, QTLoRALayer):
        for param in module.parameters():
            param.requires_grad = False 

for param in qt_lora_gpt2.gpt2_model.lm_head.grand_hypernetwork.parameters():
    param.requires_grad = True
    


print("number of training parameters in target layer(s): \n", gpt2_model.config.n_embd * gpt2_model.config.vocab_size)
print("number of parameters in the QT-GPT2: ")
n_para_lora_GPT2 = sum(p.numel() for p in qt_lora_gpt2.parameters() if p.requires_grad)
print(n_para_lora_GPT2)
print("Parameter ratio: ", 100*(n_para_lora_GPT2)/(gpt2_model.config.n_embd * gpt2_model.config.vocab_size), " %")



# Training Loop 
# Early stopping parameters
best_val_loss = float('inf')
steps_no_improve = 0

qt_lora_gpt2.cuda()

optimizer = AdamW(qt_lora_gpt2.parameters(), learning_rate )
total_steps = len(train_loader) * epochs_qt  
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

loss_list = [] 
global_step = 0 
for epoch in range(epochs_qt):
    qt_lora_gpt2.train()
    for i, batch in enumerate(train_loader):
        global_step += 1
        since_batch = time.time()
        inputs = batch['input_ids'].cuda()
        labels = inputs.clone()  # For language modeling, labels are the same as inputs
        attention_mask = batch['attention_mask'].to(device)

        with torch.cuda.amp.autocast():  # Enable mixed precision
            outputs = qt_lora_gpt2(inputs, attention_mask, labels)
            loss = outputs.loss
        loss_list.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs_qt}], Step [{i+1}/{len(train_loader)}], batch time: {time.time() - since_batch:.2f}, Loss: {loss.item()}")
        
        
    qt_lora_gpt2.eval()
    
    # Prepare the test dataset
    test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_loader = DataLoader(tokenized_test_dataset, batch_size=batch_size_test)

    # Calculate loss and perplexity on the test set
    test_loss, test_perplexity = evaluate_model(qt_lora_gpt2, test_loader)
    print(f"Test Loss: {test_loss}")
    print(f"Test Perplexity: {test_perplexity}")        
            

    qt_lora_gpt2.train()  # Switch back to training mode
