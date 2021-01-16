import os

# skipped special tokens
UNK_TOKEN = 100
CLS_TOKEN = 101
SEP_TOKEN = 102
MASK_TOKEN = 103
PAD_TOKEN = 0

# file directories
data_files_dirname = os.path.join("..", "data_files")
bert_vocab_path = os.path.join(data_files_dirname, "bert_vocab.txt")