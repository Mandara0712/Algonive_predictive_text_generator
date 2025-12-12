# src/utils.py
import re
from collections import defaultdict

def preprocess(text):
    """Lowercase and basic cleaning -> tokens list"""
    text = text.lower()
    # keep apostrophes for contractions, remove other punctuation
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    tokens = text.split()
    return tokens

def build_ngrams(tokens, n):
    """Return dict mapping ngram tuple -> count"""
    counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i+n])
        counts[key] += 1
    return counts

def top_n(counter_or_dict, k=10):
    """Return list of (item, count) sorted desc"""
    if isinstance(counter_or_dict, dict):
        items = sorted(counter_or_dict.items(), key=lambda x: -x[1])
    else:
        items = counter_or_dict.most_common(k)
    return items[:k]
