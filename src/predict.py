# src/predict.py
import os
import pickle
from collections import Counter
from utils import preprocess

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "ngrams.pkl")

class NGramPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found. Run src/build_model.py first.")
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.unigrams = Counter(data.get("unigrams", {}))
        self.bigrams = {k: v for k, v in data.get("bigrams", {}).items()}
        self.trigrams = {k: v for k, v in data.get("trigrams", {}).items()}
        self.vocab_size = len(self.unigrams) or 1

    def _trigram_candidates(self, w1, w2):
        cand = {}
        for k, v in self.trigrams.items():
            if k[0] == w1 and k[1] == w2:
                cand[k[2]] = v
        return cand

    def _bigram_candidates(self, w1):
        cand = {}
        for k, v in self.bigrams.items():
            if k[0] == w1:
                cand[k[1]] = v
        return cand

    def predict(self, text, top_k=3):
        tokens = preprocess(text)
        if not tokens:
            return [w for w,_ in self.unigrams.most_common(top_k)]
        # try trigram
        if len(tokens) >= 2:
            w1, w2 = tokens[-2], tokens[-1]
            cand = self._trigram_candidates(w1, w2)
            if cand:
                return [w for w,_ in sorted(cand.items(), key=lambda x:-x[1])[:top_k]]
        # try bigram
        w1 = tokens[-1]
        cand = self._bigram_candidates(w1)
        if cand:
            return [w for w,_ in sorted(cand.items(), key=lambda x:-x[1])[:top_k]]
        # fallback unigrams
        return [w for w,_ in self.unigrams.most_common(top_k)]

if __name__ == "__main__":
    p = NGramPredictor()
    print("predict('i love') ->", p.predict("i love"))
    print("predict('python is') ->", p.predict("python is"))
    print("predict('this project') ->", p.predict("this project"))
