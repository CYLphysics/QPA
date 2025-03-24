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



# Evaluation function to calculate loss and perplexity
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            since_batch = time.time()
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = inputs.clone().to(device)
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            print(f"Step [{i+1}/{len(data_loader)}], batch time: {time.time() - since_batch:.2f}, Loss: {loss.item()}")

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate_qubit_states_torch(n_qubit, num_vectors):
    # Calculate the total number of possible combinations
    total_combinations = 2 ** n_qubit
    if num_vectors > total_combinations:
        raise ValueError(f"Number of vectors requested ({num_vectors}) exceeds the total number of unique combinations ({total_combinations}).")
    
    # Generate random unique indices
    random_indices = random.sample(range(total_combinations), num_vectors)
    random_indices = torch.tensor(random_indices, dtype=torch.int64).unsqueeze(1)
    
    # Convert indices to binary representation and then to -1 and 1
    qubit_states = ((random_indices.unsqueeze(2) & (1 << torch.arange(n_qubit, dtype=torch.int64))) > 0).float() * 2 - 1
    
    return qubit_states

class MappingModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        # Initialize layers: an input layer, multiple hidden layers, and an output layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, X):
        X = X.type_as(self.input_layer.weight)
        X = self.input_layer(X)
        for hidden in self.hidden_layers:
            X = hidden(X)
        output = self.output_layer(X)
        return output

class QLayer(nn.Module):
    def __init__(self, n_blocks, n_qubits_):
        super().__init__()

        self.n_wires = n_qubits_
        self.n_blocks = n_blocks
        self.ry_layers = tq.QuantumModuleList()
        self.cnot_layers = tq.QuantumModuleList()

        for _ in range(self.n_blocks):
            self.ry_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.cnot_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CNOT,
                    n_wires=self.n_wires,
                    has_params=False,
                    trainable=False,
                    circular=False,
                )
            )
            
    def forward(self):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=1, device=next(self.parameters()).device
        )
        device=next(self.parameters()).device
        easy_scale_coeff = 2**(self.n_wires-1)
        gamma = 0.1
        beta  = 0.60
        alpha = 0.15
        for k in range(self.n_blocks):
            self.ry_layers[k](qdev)
            self.cnot_layers[k](qdev)
            
        state_mag = qdev.get_states_1d().abs()[0] 
        x = torch.abs(state_mag) ** 2
        x = x.reshape(2**(self.n_wires),1)
        x = (beta*torch.tanh(gamma*easy_scale_coeff*x))**(alpha) 
        x = x - torch.mean(x)
        x.to(device)
        return x    

class MultiLayerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(MultiLayerAttention, self).__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads) for _ in range(num_layers)])
        
    def forward(self, x):
        attn_output = x
        for layer in self.layers:
            attn_output, _ = layer(attn_output, attn_output, attn_output)
        return attn_output
    
class QT_HyperNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_sub_hypernetwork, chunk_size, qnn_depth):
        super(QT_HyperNet, self).__init__()
        
        self.n_sub_hypernetwork = n_sub_hypernetwork
        self.weight_length = int(np.ceil((vocab_size * hidden_size) / self.n_sub_hypernetwork ))

        self.out_dim_MPS = 32
        self.out_dim_MLP = chunk_size
        self.batch_size = int(np.ceil((self.weight_length/self.out_dim_MLP))) 
        self.dropout = nn.Dropout(p=0.)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_mapping = "MLP"
        self.classical_layers = "MLP"

        
        self.n_qubit_qt = int(np.ceil(np.log2(self.batch_size)))
        self.n_qubit = self.n_qubit_qt
        self.q_depth    = qnn_depth
        self.QuantumNN = QLayer(self.q_depth, self.n_qubit_qt).to(self.device)   

        if self.init_mapping == "MLP":
            self.MappingNetwork = MappingModel(self.n_qubit+1, [32, 64, 128, 128, 64, 32], self.out_dim_MPS)
            
            
        if self.classical_layers == "MLP": 
            self.fc1 = nn.Linear(self.out_dim_MPS, self.out_dim_MLP)


    def forward(self):
        
        compute_method = "checkpoint"
                
        
        probs_ = self.QuantumNN().flatten()
        probs_ = probs_[:self.batch_size]
        probs_ = probs_.reshape(self.batch_size, 1, 1)

        
        qubit_states_torch = generate_qubit_states_torch(self.n_qubit, self.batch_size)[:self.weight_length].to(self.device)
        combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=2)
        if self.init_mapping == "MPS":
            combined_data_torch = combined_data_torch.reshape(self.batch_size,  self.n_qubit + 1)


        prob_val_post_processed_list = []
        if compute_method == "checkpoint":

            batch_data = combined_data_torch[0:self.batch_size]
            batch_data.requires_grad_()
            
            prob_val_post_processed_batch = checkpoint(self.MappingNetwork, batch_data)

            if self.classical_layers == "MLP":
                
                prob_val_post_processed_batch = checkpoint(self.dropout, prob_val_post_processed_batch)
                prob_val_post_processed_batch = checkpoint(self.fc1, prob_val_post_processed_batch)
                              
            
            prob_val_post_processed_list.append(prob_val_post_processed_batch)
                      
            torch.cuda.empty_cache()

        prob_val_post_processed_list = prob_val_post_processed_list[:self.weight_length]
        prob_val_post_processed = torch.cat(prob_val_post_processed_list, dim=0)
        
        prob_val_post_processed = prob_val_post_processed.view(-1)[:self.weight_length]
        prob_val_post_processed = prob_val_post_processed - prob_val_post_processed.mean()
        
        torch.cuda.empty_cache()

        return prob_val_post_processed

    
    
class QPA_LoRALayer(nn.Module):
    def __init__(self, original_layer, r, alpha, n_sub_hypernetwork, chunk_size, qnn_depth):
        super(QPA_LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(p=0.05)
        self.dtype = torch.float32  # Ensure all tensors are of this type

        # Generate the parameters of A and B
        self.grand_hypernetwork = nn.ModuleList([
            QT_HyperNet(
                original_layer.weight.size(0)*r + r*original_layer.weight.size(1),  
                1, 
                n_sub_hypernetwork, chunk_size, qnn_depth)
            for _ in range(n_sub_hypernetwork)]).cuda()
        
        # Freeze the original layer's parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        gen_weights = []
        for sub_hypernetwork in self.grand_hypernetwork:
            gen_weights.append(sub_hypernetwork())
        self.generated_weights = torch.cat(gen_weights, dim=0).view(-1)[:self.original_layer.weight.size(0)*self.r + self.r*self.original_layer.weight.size(1)].cuda()
        self.generated_weights_A = self.generated_weights[:self.original_layer.weight.size(0)*self.r].view(self.original_layer.weight.size(0), self.r).type(self.dtype)
        self.generated_weights_B = self.generated_weights[self.original_layer.weight.size(0)*self.r:].view(self.r, self.original_layer.weight.size(1)).type(self.dtype)

        batch_size, seq_len, hidden_size = x.size()
        x_reshaped = self.dropout(x).view(-1, hidden_size)
        delta = (x_reshaped @ self.generated_weights_B.t()) @ self.generated_weights_A.t()
        delta = delta * (self.alpha / self.r)
        delta = delta.view(batch_size, seq_len, self.generated_weights_A.shape[0])
        return self.original_layer(x) + delta      
    
    
class QPA_LoRAGPT2(nn.Module):
        
    def __init__(self, gpt2_model, n_sub_hypernetwork, LoRA_rank, chunk_size, qnn_depth):
        super(QPA_LoRAGPT2, self).__init__()

        self.gpt2_model = gpt2_model
        
        for name, module in self.gpt2_model.named_modules():
            if name == 'lm_head':
                lora_layer = QPA_LoRALayer(
                    module,
                    r=LoRA_rank,
                    alpha=LoRA_rank*2,
                    n_sub_hypernetwork=n_sub_hypernetwork,
                    chunk_size=chunk_size,
                    qnn_depth=qnn_depth)  
                setattr(self.gpt2_model, name, lora_layer)
            
    def forward(self, input_ids, attention_mask, labels):

        outputs = self.gpt2_model(input_ids, attention_mask=attention_mask, labels = labels)

        
        return outputs
    
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r, alpha):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(p=0.0)
        self.A = nn.Parameter(torch.randn(original_layer.weight.size(0), r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, original_layer.weight.size(1)) * 0.01)


        # Freeze the original layer's parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        batch_size, seq_len, hidden_size = x.size()
        x_reshaped = self.dropout(x).view(-1, hidden_size)
        delta = (x_reshaped @ self.B.t()) @ self.A.t()
        delta = delta * (self.alpha / self.r)
        delta = delta.view(batch_size, seq_len, self.A.shape[0])
        return self.original_layer(x) + delta    
    
    
class LoRAGPT2(nn.Module):
        
    def __init__(self, gpt2_model):
        super(LoRAGPT2, self).__init__()

        self.gpt2_model = gpt2_model
        
        # Freeze all parameters in the gpt2_model
        for param in self.gpt2_model.parameters():
            param.requires_grad = True 
            
        # Apply LoRA to specific layers
        
        for name, module in self.gpt2_model.named_modules():
            if name == 'lm_head':
                lora_layer = LoRALayer(module, r=LoRA_rank, alpha = LoRA_rank*2) 
                setattr(self.gpt2_model, name, lora_layer)
            
    def forward(self, input_ids, attention_mask, labels):

        outputs = self.gpt2_model(input_ids, attention_mask=attention_mask, labels = labels)
        
        return outputs
        
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)