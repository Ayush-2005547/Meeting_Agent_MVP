import json
from datasets import load_dataset

# Load first 10 dialogues from DialogSum
ds = load_dataset("knkarthick/dialogsum", split="train[:10]")

# Prepare the sample_meetings.json structure
sample_meetings = []

for i, item in enumerate(ds):
    sample_meetings.append({
        "meeting_id": str(i+1),
        "reference": item['summary'],
        "generated": "", 
    })

DATA_PATH = "../data/sample_meetings.json"   # points to JSON
OUTPUT_PATH = "../outputs/sample_meetings_output.json"  # wherever you want results

# Save to JSON
with open("../data/sample_meetings.json", "w", encoding="utf-8") as f:
    json.dump(sample_meetings, f, indent=2)

print("sample_meetings.json created with 10 DialogSum dialogues.")
