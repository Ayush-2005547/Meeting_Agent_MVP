import os
import json
import re
from tqdm import tqdm
from evaluate import load
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "summaries_subset_optimized.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "evaluated_summaries.json")

# -------------------------------
# Load data
# -------------------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} summaries for evaluation.")

# -------------------------------
# Setup metrics
# -------------------------------
rouge = load("rouge")

def extract_keywords(text, top_k=10):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text])
    indices = np.argsort(tfidf.toarray()).flatten()[::-1][:top_k]
    features = np.array(vectorizer.get_feature_names_out())
    return features[indices].tolist()

# -------------------------------
# Evaluate all
# -------------------------------
results = []

for item in tqdm(data, desc="Evaluating summaries"):
    ref = item.get("reference_summary", "").strip()
    gen = item.get("generated_summary", "").strip()

    if not ref or not gen:
        continue

    # Compute ROUGE
    scores = rouge.compute(predictions=[gen], references=[ref])
    # Updated for new API (floats instead of objects)
    r1 = scores["rouge1"]
    rL = scores["rougeL"]

    # Lengths
    ref_len = len(ref.split())
    gen_len = len(gen.split())
    ratio = round(gen_len / max(1, ref_len), 2)

    # Keywords
    ref_kw = extract_keywords(ref)
    gen_kw = extract_keywords(gen)
    missing = [w for w in ref_kw if w not in gen_kw]

    results.append({
        "meeting_id": item.get("meeting_id", ""),
        "reference": ref,
        "generated_summary": gen,
        "ref_len": ref_len,
        "gen_len": gen_len,
        "length_ratio": ratio,
        "rouge1_f1": round(r1, 3),
        "rougeL_f1": round(rL, 3),
        "ref_keywords": ref_kw,
        "gen_keywords": gen_kw,
        "missing_keywords": missing
    })

# -------------------------------
# Save results
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"✅ Evaluation complete! Saved to {OUTPUT_PATH}")
