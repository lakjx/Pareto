import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import time
import math
from tqdm import tqdm
import os
from collections import Counter

# --- 1. 全局配置 ---
CONFIG = {
    "epochs": 40,
    "batch_size": 20,
    "eval_batch_size": 10,
    "learning_rate": 0.001,
    "seq_len": 35,
    "d_model": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "clip_grad": 0.25,
    "seed": 1568
}

# --- 2. 工具函数 ---
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 3. 自定义词汇表构建 (替代torchtext.vocab) ---
class Vocabulary:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.word2idx = {}
        self.idx2word = []
        self.unk_token = "<unk>"
        self.pad_token = "<pad>" # 虽然PTB不常用，但定义一个总是好的

    def build_vocab(self, text_iterator, min_freq=1):
        """从文本迭代器构建词汇表"""
        counter = Counter()
        for text in tqdm(text_iterator, desc="Building vocab"):
            counter.update(self.tokenizer(text))
        
        # 添加特殊符号
        self.idx2word.extend([self.pad_token, self.unk_token])

        # 根据最小词频筛选并添加单词
        for word, freq in counter.items():
            if freq >= min_freq:
                self.idx2word.append(word)
        
        # 创建 word -> index 的映射
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.unk_index = self.word2idx[self.unk_token]

    def numericalize(self, text):
        """将文本转换为ID序列"""
        tokens = self.tokenizer(text)
        ids = [self.word2idx.get(token, self.unk_index) for token in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.idx2word)

# --- 4. 数据处理 (替代torchtext.datasets) ---
def get_data_and_vocab():
    print("Loading PennTreebank dataset from Hugging Face...")
    # 使用datasets库加载数据集
    raw_datasets = load_dataset("ptb_text_only", 
                                "penn_treebank", 
                                trust_remote_code=True,
                                cache_dir='pareto_exp/cache'
    )
    
    # 简单的空格分词器
    tokenizer = lambda text: text.split()

    # 构建词汇表
    vocab = Vocabulary(tokenizer)
    # 只在训练集上构建词汇表
    vocab.build_vocab(raw_datasets['train']['sentence'])

    # 数值化所有数据集
    print("Numericalizing data...")
    train_data = torch.cat([vocab.numericalize(text) for text in raw_datasets['train']['sentence']])
    val_data = torch.cat([vocab.numericalize(text) for text in raw_datasets['validation']['sentence']])
    test_data = torch.cat([vocab.numericalize(text) for text in raw_datasets['test']['sentence']])
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train data length (tokens): {len(train_data)}")

    return train_data, val_data, test_data, vocab

def batchify(data, bsz, device):
    num_batches = data.size(0) // bsz
    data = data.narrow(0, 0, num_batches * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# --- 5. 模型定义 (与之前验证过的RNNModel一致) ---
class RNNModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, dropout=dropout, batch_first=False)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.decoder.weight = self.embedding.weight
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, hidden):
        emb = self.drop(self.embedding(src))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.d_model),
                weight.new_zeros(self.num_layers, bsz, self.d_model))


# --- 6. 训练与评估流程 ---
def train_one_epoch(model, train_data, criterion, optimizer, scheduler, vocab_size, epoch):
    model.train()
    total_loss = 0.
    hidden = model.init_hidden(CONFIG['batch_size'])
    
    pbar = tqdm(range(0, train_data.size(0) - 1, CONFIG['seq_len']), desc=f"Epoch {epoch}/{CONFIG['epochs']}")
    for i in pbar:
        data, targets = get_batch(train_data, i, CONFIG['seq_len'])
        hidden = tuple([h.detach() for h in hidden])
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_grad'])
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    scheduler.step()

def evaluate(model, eval_data, criterion, vocab_size):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(CONFIG['eval_batch_size'])
    
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, CONFIG['seq_len']):
            data, targets = get_batch(eval_data, i, CONFIG['seq_len'])
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, vocab_size), targets)
            total_loss += len(data) * loss.item()
    
    avg_loss = total_loss / (len(eval_data) - 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# --- 7. 主函数 ---
def main():
    set_random_seed(CONFIG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, val_data, test_data, vocab = get_data_and_vocab()
    
    train_data = batchify(train_data, CONFIG['batch_size'], device)
    val_data = batchify(val_data, CONFIG['eval_batch_size'], device)
    test_data = batchify(test_data, CONFIG['eval_batch_size'], device)

    vocab_size = len(vocab)
    model = RNNModel(
        vocab_size, CONFIG['d_model'], CONFIG['num_layers'], CONFIG['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)

    print("\n--- Starting Training on PennTreebank (No TorchText) ---")
    best_val_loss = float('inf')
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_start_time = time.time()
        train_one_epoch(model, train_data, criterion, optimizer, scheduler, vocab_size, epoch)
        val_loss, val_ppl = evaluate(model, val_data, criterion, vocab_size)
        elapsed = time.time() - epoch_start_time
        
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ptb_model_no_torchtext.pt')
            print("Saved new best model.")

    model.load_state_dict(torch.load('best_ptb_model_no_torchtext.pt'))
    test_loss, test_ppl = evaluate(model, test_data, criterion, vocab_size)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}')
    print('=' * 89)

if __name__ == '__main__':
    main()