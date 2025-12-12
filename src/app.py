# src/app.py
from flask import Flask, render_template, request, jsonify
from predict import NGramPredictor
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"), static_folder=os.path.join(os.path.dirname(__file__), "..", "static"))
predictor = None

try:
    predictor = NGramPredictor()
except Exception as e:
    # predictor will be None if model not built yet
    print("Warning: predictor not ready:", e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    text = data.get("text", "")
    k = int(data.get("k", 3))
    if predictor is None:
        return jsonify({"error": "Model not found. Run src/build_model.py and restart the app."}), 400
    suggestions = predictor.predict(text, top_k=k)
    return jsonify({"suggestions": suggestions})

@app.route("/predict", methods=["POST"])
def form_predict():
    text = request.form.get("text", "")
    if predictor is None:
        return render_template("index.html", error="Model not built. Run src/build_model.py.")
    suggestions = predictor.predict(text, top_k=5)
    return render_template("index.html", text=text, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
