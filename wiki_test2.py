import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import time
import math
from tqdm import tqdm

# --- 1. 全局配置 ---
CONFIG = {
    "epochs": 20,                   # 减少epochs，先看收敛情况
    "batch_size": 256,               # 减小batch size，有助于稳定训练
    "learning_rate": 0.001,         # 适中的学习率
    "seq_len": 35,                  # 使用标准的序列长度
    "tokenizer_path": "pareto_exp/wikitext-tokenizer",
    "train_samples_to_use": 1e8,  # 先用更小的数据集测试
    "eval_samples_to_use": 1e8,    
    "seed": 1568,
    "clip_grad": 0.25,
    "dropout": 0.4,
    "embed_size": 128,              # 嵌入维度
    "hidden_size": 64,             # 隐藏层维度
    "num_layers": 4
}

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WikiTextDataset(Dataset):
    def __init__(self, tokenized_data, tokenizer, seq_len=35):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.all_tokens = []
        for example in tokenized_data:
            if example['input_ids']:
                self.all_tokens.extend(example['input_ids'])
    
    def __len__(self):
        return len(self.all_tokens) - self.seq_len - 1
    
    def __getitem__(self, idx):
        input_seq = self.all_tokens[idx : idx + self.seq_len]
        target_seq = self.all_tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# --- 修复后的RNN模型 ---
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           dropout=dropout if num_layers > 1 else 0, 
                           batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # 修复权重绑定条件
        if embed_size == hidden_size:
            self.decoder.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.drop(out)
        decoded = self.decoder(out)
        return decoded, hidden
    
    def init_hidden(self, bsz, device):
        # 修复：在指定设备上初始化隐藏状态
        return (torch.zeros(self.num_layers, bsz, self.hidden_size, device=device),
                torch.zeros(self.num_layers, bsz, self.hidden_size, device=device))

def load_data(tokenizer_path, seq_len, train_samples, eval_samples):
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize_function(examples):
        return tokenizer(examples['text'])

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    full_train_dataset = WikiTextDataset(tokenized_datasets['train'], tokenizer, seq_len)
    full_test_dataset = WikiTextDataset(tokenized_datasets['test'], tokenizer, seq_len)

    num_samples_for_train = min(train_samples, len(full_train_dataset))
    train_indices = range(num_samples_for_train)
    train_dataset = Subset(full_train_dataset, train_indices)
    
    num_samples_for_eval = min(eval_samples, len(full_test_dataset))
    eval_indices = range(num_samples_for_eval)
    test_dataset = Subset(full_test_dataset, eval_indices)

    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Using {len(train_dataset)} samples for training.")
    print(f"Using {len(test_dataset)} samples for evaluation.")
    
    return train_dataset, test_dataset, tokenizer

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(dataloader, desc="Evaluating")):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # 为每个batch重新初始化隐藏状态
            hidden = model.init_hidden(batch_size, device)
            
            output, hidden = model(data, hidden)
            output_reshaped = output.view(-1, model.vocab_size)
            target_reshaped = target.view(-1)
            
            loss = criterion(output_reshaped, target_reshaped)
            total_loss += loss.item()
            total_tokens += target_reshaped.numel()
            
    avg_loss = total_loss / len(dataloader)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def main():
    print("--- Starting RNN Training ---")
    print(f"Configuration: {CONFIG}")
    
    set_random_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset, test_dataset, tokenizer = load_data(
        CONFIG["tokenizer_path"], CONFIG["seq_len"], 
        CONFIG["train_samples_to_use"], CONFIG["eval_samples_to_use"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
    
    model = RNNModel(
        vocab_size=tokenizer.vocab_size,
        embed_size=CONFIG["embed_size"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    print("\n--- Training Start ---")
    best_val_loss = float('inf')

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        
        # 修复：正确的隐藏状态管理
        for i, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # 每个batch开始时初始化隐藏状态
            if i == 0:
                hidden = model.init_hidden(batch_size, device)
            else:
                # 分离隐藏状态，防止梯度在batch间传播
                hidden = tuple(h.detach() for h in hidden)
                # 如果batch size改变，重新初始化
                if hidden[0].size(1) != batch_size:
                    hidden = model.init_hidden(batch_size, device)
            
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            
            output_reshaped = output.view(-1, tokenizer.vocab_size)
            target_reshaped = target.view(-1)
            loss = criterion(output_reshaped, target_reshaped)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_ppl = evaluate(model, test_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
        
        scheduler.step()
        print("-" * 70)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()