# Predictive Text Generator (N-gram model)

## Overview
A simple Python-based Predictive Text Generator using n-gram language modeling (trigram → bigram → unigram backoff) with a Flask demo UI.

## Setup
1. Create a virtualenv (recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Place your corpus in `data/sample_corpus.txt` (default provided).

## Build model
```
python src/build_model.py
```
Or run as a module from the project root:

```
python -m src.build_model
```
This creates `model/ngrams.pkl`.

## Run demo (Flask)
```
python src/app.py
```
Or run as a module:

```
python -m src.app
```
Open http://127.0.0.1:5000 in your browser.

UI: A modern responsive interface is available at the root (`/`) with live suggestions and clickable suggestion chips.
The UI now includes a "My Predictions" panel that logs your chosen suggestions. Click a suggestion to append it to your text and the choice will be saved to your browser (localStorage), with a "Clear" button to remove history.

## How it works (brief)
- Preprocess: lowercase + basic cleaning.
- Build counts for unigrams, bigrams, trigrams.
- Prediction: given input text, try trigram candidates for last 2 words; if none, try bigrams; else fallback to top unigrams.

## Files
- `src/build_model.py` — build/save n-gram counts
- `src/predict.py` — NGramPredictor class
- `src/app.py` — Flask demo
- `templates/index.html` — demo UI
- `static/style.css` — styling

## Extensions & Improvements
- Add smoothing (e.g., add-1) and probability calculation.
- Add user custom dictionary to boost suggestions.
- Add token-level autocomplete and spell correction.
- Replace or augment with lightweight ML (Keras LSTM or transformer) for better results.
