# scripts/load_preprocess.py

from datasets import load_dataset
import os
import json
from tqdm import tqdm

# -------------------------------
# 1. Define paths
# -------------------------------
# Ensure outputs/ folder is created inside the project folder
DATA_PATH = os.path.join(os.getcwd(), "data")
OUTPUT_PATH = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -------------------------------
# 2. Load the MeetingBank dataset
# -------------------------------
dataset = load_dataset("lytang/MeetingBank-transcript")

# Inspect dataset structure
print(dataset)
print(dataset['train'][0])  # First training example

# -------------------------------
# 3. Preprocessing function
# -------------------------------
def preprocess_transcript(text):
    """
    Cleans a transcript:
    - Removes line breaks
    - Removes extra spaces
    """
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    return text

# -------------------------------
# 4. Preprocess all transcripts
# -------------------------------
cleaned_data = []
for item in tqdm(dataset['train']):
    # 'source' contains the meeting transcript
    transcript = preprocess_transcript(item['source'])

    # 'reference' contains the summary / minutes (may be empty)
    summary = item.get('reference', "")

    # Store cleaned transcript + summary
    cleaned_data.append({
        "meeting_id": item.get('meeting_id', ""),
        "transcript": transcript,
        "summary": summary
    })

# -------------------------------
# 5. Save cleaned transcripts
# -------------------------------
output_file = os.path.join(OUTPUT_PATH, "meetingbank_train_cleaned.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

print(f"✅ Cleaned MeetingBank transcripts saved at: {output_file}")
