# scripts/summarize_fast_optimized.py

import os
import json
import random
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from datasets import Dataset
from collections import defaultdict

# -------------------------------
# 1. Configurable parameters
# -------------------------------
INPUT_PATH = os.path.join(os.getcwd(), "outputs", "meetingbank_train_cleaned.json")
OUTPUT_PATH = os.path.join(os.getcwd(), "outputs", "summaries_subset_optimized.json")

SUBSET_SIZE = 200        # Number of meetings to process
BATCH_SIZE = 32          # Larger batch size for GPU speed
CHUNKS_PER_MEETING = 5   # Max chunks per meeting (before merging)
MIN_CHUNK_WORDS = 50     # Merge chunks smaller than this

USE_DISTILBART = True    # Faster distilled BART for prototyping

# -------------------------------
# 2. Load cleaned transcripts
# -------------------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} transcripts.")

# -------------------------------
# 3. Take a random subset of meetings
# -------------------------------
subset_data = random.sample(data, min(SUBSET_SIZE, len(data)))
print(f"Processing subset of {len(subset_data)} meetings.")

# -------------------------------
# 4. Load summarization pipeline & tokenizer
# -------------------------------
model_name = "sshleifer/distilbart-cnn-12-6" if USE_DISTILBART else "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name, device=0)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------------------
# 5. Chunking function with merging small chunks
# -------------------------------
def chunk_text(text, max_words=500, min_words=MIN_CHUNK_WORDS):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(i + max_words, len(words))
        chunk = words[i:end]
        # Merge with previous if too small
        if chunks and len(chunk) < min_words:
            chunks[-1].extend(chunk)
        else:
            chunks.append(chunk)
        i = end
    return [" ".join(c) for c in chunks]

# -------------------------------
# 6. Prepare all chunks
# -------------------------------
all_items = []
for item in tqdm(subset_data):
    transcript = item['transcript']
    reference = item.get('summary', "")
    chunks = chunk_text(transcript)[:CHUNKS_PER_MEETING]
    for chunk in chunks:
        all_items.append({
            "meeting_id": item.get('meeting_id', ""),
            "chunk": chunk,
            "reference": reference
        })

dataset = Dataset.from_list(all_items)
print(f"Total chunks to summarize: {len(dataset)}")

# -------------------------------
# 7. Summarize in batch mode with dynamic max_length
# -------------------------------
summarized_chunks = []

for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch = dataset[i: i + BATCH_SIZE]["chunk"]
    input_lengths = [len(chunk.split()) for chunk in batch]
    batch_max_length = max(40, min(int(max(input_lengths) * 1.5), 120))
    try:
        summaries = summarizer(
            batch,
            max_length=batch_max_length,
            min_length=20,
            do_sample=False
        )
        summarized_chunks.extend(summaries)
    except Exception as e:
        print(f"Skipping a batch due to error: {e}")

# -------------------------------
# 8. Aggregate summaries by meeting_id
# -------------------------------
final_summaries = defaultdict(lambda: {"transcript": "", "generated_summary": "", "reference_summary": ""})

for idx, item in enumerate(all_items):
    mid = item["meeting_id"]
    final_summaries[mid]["generated_summary"] += summarized_chunks[idx]['summary_text'] + " "
    final_summaries[mid]["transcript"] = final_summaries[mid]["transcript"] or item["chunk"]
    final_summaries[mid]["reference_summary"] = item["reference"]

final_output = []
for mid, content in final_summaries.items():
    final_output.append({
        "meeting_id": mid,
        "transcript": content["transcript"],
        "generated_summary": content["generated_summary"].strip(),
        "reference_summary": content["reference_summary"]
    })

# -------------------------------
# 9. Save summaries
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)

print(f"✅ Optimized summaries saved at {OUTPUT_PATH}")

# -------------------------------
# 10. Preview 3 sample summaries with diff
# -------------------------------
import difflib

def highlight_diff(ref, gen):
    ref_words = ref.split()
    gen_words = gen.split()
    diff = difflib.ndiff(ref_words, gen_words)
    return ' '.join(diff)

print("\n--- Sample summaries with diff ---")
for sample in final_output[:3]:
    print(f"\nMeeting ID: {sample['meeting_id']}")
    print("Reference Summary:")
    print(sample['reference_summary'])
    print("\nGenerated Summary:")
    print(sample['generated_summary'])
    print("\nDiff Highlight:")
    print(highlight_diff(sample['reference_summary'], sample['generated_summary']))
