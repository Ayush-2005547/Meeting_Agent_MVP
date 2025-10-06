import json

DATA_PATH = "../data/sample_meetings.json"
OUTPUT_PATH = "../outputs/evaluated_summaries.json"

def get_keywords(text, top_n=5):
    stopwords = set(["the","and","a","an","is","in","of","to","for","on","with","that"])
    words = [w.lower() for w in text.split() if w.lower() not in stopwords]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [w for w, _ in top_words]

# Load JSON
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

for item in dataset:
    ref = item.get("reference", "")
    gen = item.get("generated", "")
    results.append({
        "meeting_id": item.get("meeting_id", ""),
        "ref_len": len(ref.split()),
        "gen_len": len(gen.split()),
        "ref_keywords": get_keywords(ref),
        "gen_keywords": get_keywords(gen),
        "missing_keywords": [kw for kw in get_keywords(ref) if kw not in get_keywords(gen)]
    })

# Save to outputs
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Evaluation done, Results saved to {OUTPUT_PATH}")
