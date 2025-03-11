import numpy as np
import re


def tokenizer(text, vocab, max_len=150):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    tokens = text.split()
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokens[:max_len]]
    encodec_padded = np.pad(encoded, (0, max_len - len(encoded)), constant_values=vocab["<PAD>"])[:max_len]
    encodec_padded = encodec_padded.reshape(1, -1)
    return encodec_padded
