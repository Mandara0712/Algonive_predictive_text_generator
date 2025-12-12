# src/build_model.py
import os
import pickle
from collections import Counter
from utils import preprocess, build_ngrams

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_corpus.txt")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "ngrams.pkl")

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_and_save():
    text = load_corpus(DATA_PATH)
    tokens = preprocess(text)
    unigrams = Counter(tokens)
    bigrams = build_ngrams(tokens, 2)
    trigrams = build_ngrams(tokens, 3)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model = {
        "unigrams": dict(unigrams),
        "bigrams": {k:v for k,v in bigrams.items()},
        "trigrams": {k:v for k,v in trigrams.items()}
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")
    print("Sample unigrams:", list(unigrams.items())[:10])

if __name__ == "__main__":
    build_and_save()
