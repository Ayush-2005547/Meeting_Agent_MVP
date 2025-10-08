import os
import json
from collections import Counter
from rouge_score import rouge_scorer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_meetings.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "evaluated_summaries.json")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Stopwords
STOPWORDS = set(["the","and","a","an","is","in","of","to","for","on","with","that"])

def get_keywords(text, top_n=5):
    words = [w.lower() for w in text.split() if w.lower() not in STOPWORDS]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)

results = []

for item in dataset:
    ref = item.get("reference", "")
    gen = item.get("generated", "")

    ref_keywords = get_keywords(ref)
    gen_keywords = get_keywords(gen)
    missing_keywords = [kw for kw in ref_keywords if kw not in gen_keywords]

    rouge_scores = scorer.score(ref, gen)
    len_ratio = len(gen.split()) / max(len(ref.split()), 1)

    results.append({
        "meeting_id": item.get("meeting_id", ""),
        "reference": ref,
        "generated_summary": gen,
        "ref_len": len(ref.split()),
        "gen_len": len(gen.split()),
        "length_ratio": len_ratio,
        "ref_keywords": ref_keywords,
        "gen_keywords": gen_keywords,
        "missing_keywords": missing_keywords,
        "rouge1_f1": rouge_scores['rouge1'].fmeasure,
        "rougeL_f1": rouge_scores['rougeL'].fmeasure
    })

# Save results
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Evaluation done, Results saved to {OUTPUT_PATH}")
