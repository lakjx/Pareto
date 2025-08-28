import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import time
from tqdm import tqdm # 用于显示美观的进度条
import math # 需要引入 math 库

# --- 1. 全局配置 ---
CONFIG = {
    "epochs": 20,                   # 训练周期可以适当增加，因为数据集变小了
    "batch_size": 256,              # 批处理大小
    "learning_rate": 0.0001,        # 关键：使用推荐的较低学习率
    "seq_len": 64,                  # 序列长度
    "tokenizer_path": "pareto_exp/wikitext-tokenizer", # 您本地分词器的路径
    "train_samples_to_use": 100000, # 新增：用于训练的样本数量，大大减少训练量
    "eval_samples_to_use": 20000,   # 用于评估的测试集样本数
    "seed": 1568                    # 随机种子
}

# --- 2. 模型和数据集定义 (与之前一致) ---

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WikiTextDataset(Dataset):
    """ 高效的Dataset实现，动态生成样本 """
    def __init__(self, tokenized_data, tokenizer, seq_len=64):
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.squeeze(1) # 形状变为 [max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # 将位置编码加到输入张量上
        # x.size(1) 是序列的长度 seq_len
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

# 修改后的 SimpleTransformer 类
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, seq_len=64):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 使用新的、标准的正弦位置编码模块
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        
        # 将词嵌入向量传入位置编码模块
        embedded_with_pos = self.pos_encoder(embedded)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # 将带有位置信息的向量传入Transformer
        encoded = self.transformer(embedded_with_pos, mask=mask)
        
        output = self.fc_out(encoded)
        return output

# --- 3. 数据加载与评估函数 ---

def load_data(tokenizer_path, seq_len, train_samples, eval_samples):
    """加载和预处理数据，并创建指定大小的子集"""
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

    # 创建完整的Dataset对象
    full_train_dataset = WikiTextDataset(tokenized_datasets['train'], tokenizer, seq_len)
    full_test_dataset = WikiTextDataset(tokenized_datasets['test'], tokenizer, seq_len)

    # --- 关键改动：从完整数据集中创建指定大小的训练子集 ---
    num_samples_for_train = min(train_samples, len(full_train_dataset))
    train_indices = range(num_samples_for_train)
    train_dataset = Subset(full_train_dataset, train_indices)
    
    # 创建用于评估的测试子集
    num_samples_for_eval = min(eval_samples, len(full_test_dataset))
    eval_indices = range(num_samples_for_eval)
    test_dataset = Subset(full_test_dataset, eval_indices)

    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Using {len(train_dataset)} samples for training.")
    print(f"Using {len(test_dataset)} samples for evaluation.")
    
    return train_dataset, test_dataset, tokenizer

def evaluate(model, dataloader, criterion, device):
    """评估模型性能"""
    # ... (这部分代码和之前完全一样，为简洁省略) ...
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_reshaped = output.view(-1, output.size(-1))
            target_reshaped = target.view(-1)
            loss = criterion(output_reshaped, target_reshaped)
            total_loss += loss.item()
            _, predicted = torch.max(output_reshaped, 1)
            mask = target_reshaped != criterion.ignore_index
            total_correct += (predicted[mask] == target_reshaped[mask]).sum().item()
            total_tokens += mask.sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    ppl = np.exp(avg_loss)
    return avg_loss, accuracy, ppl

# --- 4. 主训练流程 ---

def main():
    """主函数，执行训练和评估"""
    # ... (这部分代码和之前几乎一样，为简洁省略) ...
    print("--- Starting Centralized Training on a SUBSET ---")
    print(f"Configuration: {CONFIG}")
    set_random_seed(CONFIG["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataset, test_dataset, tokenizer = load_data(
        CONFIG["tokenizer_path"], CONFIG["seq_len"], CONFIG["train_samples_to_use"], CONFIG["eval_samples_to_use"]
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
    model = SimpleTransformer(vocab_size=tokenizer.vocab_size, seq_len=CONFIG["seq_len"]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    print("\n--- Training Start ---")


    single_batch_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"])
    the_only_batch = next(iter(single_batch_loader)) 
    # 创建一个只包含这一个批次的“伪”数据加载器
    # 我们将用这个批次的数据反复进行训练和“评估”
    overfit_loader = [the_only_batch] 

    print("\n--- Sanity Check: Overfitting a single batch ---")
    for epoch in range(50): # 训练更多次，观察loss变化
        model.train()
        epoch_loss = 0
        
        # 只在这个固定的小批量数据上训练
        for data, target in overfit_loader: # 循环只会执行一次
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output_reshaped = output.view(-1, output.size(-1))
            target_reshaped = target.view(-1)
            loss = criterion(output_reshaped, target_reshaped)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 在同一个小批量数据上评估
        val_loss, val_acc, val_ppl = evaluate(model, overfit_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/50 | "
            f"Loss on single batch: {val_loss:.4f} | "
            f"Accuracy: {val_acc:.4f} | "
            f"PPL: {val_ppl:.2f}")


    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output_reshaped = output.view(-1, output.size(-1))
            target_reshaped = target.view(-1)
            loss = criterion(output_reshaped, target_reshaped)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc, val_ppl = evaluate(model, test_loader, criterion, device)
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_acc:.4f} | "
              f"Val PPL: {val_ppl:.2f}")
        print("-" * 60)

if __name__ == '__main__':
    main()