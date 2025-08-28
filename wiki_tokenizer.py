# train_tokenizer.py
from tokenizers import BertWordPieceTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import BertTokenizerFast

# Load the raw dataset to train on
raw_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

# Create an iterator over the text
def get_training_corpus():
    for i in range(0, len(raw_dataset), 1000):
        yield raw_dataset[i : i + 1000]["text"]

# 1. Initialize a new tokenizer
# We'll use WordPiece, the same type used by BERT.
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True, # Match the old 'basic_english' tokenizer behavior
)

# 2. Train the tokenizer
# You can choose your desired vocabulary size. Let's pick 10,000.
tokenizer.train_from_iterator(
    get_training_corpus(),
    vocab_size=20000, 
    min_frequency=2, # Words must appear at least twice
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)

# 3. Save the tokenizer
import os
os.makedirs("pareto_exp/wikitext-tokenizer", exist_ok=True)
tokenizer.save_model("pareto_exp/wikitext-tokenizer") 
print("Tokenizer trained and saved to ./wikitext-tokenizer")

vocab_file = "pareto_exp/wikitext-tokenizer/vocab.txt"
tokenizer = BertTokenizerFast(vocab_file=vocab_file, lowercase=True)
tokenizer.save_pretrained("pareto_exp/wikitext-tokenizer")