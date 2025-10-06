# transcribe.py
import os
import json
import whisper

# -------------------------
# Paths
# -------------------------
AUDIO_FOLDER = "../audios"             # Folder containing WAV files
OUTPUT_JSON = "../data/sample_meetings.json"  # Output JSON

# -------------------------
# Load Whisper model
# -------------------------
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded.")

# -------------------------
# Transcribe all WAV files
# -------------------------
results = []

for filename in os.listdir(AUDIO_FOLDER):
    if filename.lower().endswith(".wav"):
        audio_path = os.path.join(AUDIO_FOLDER, filename)
        print(f"Transcribing: {audio_path} ...")
        
        result = model.transcribe(audio_path)
        text = result["text"].strip()
        
        # Save per-audio transcription
        results.append({
            "meeting_id": filename,
            "generated": text,
            "reference": ""  # can be filled later
        })
        print(f"Done: {filename}\n{text}\n")

# -------------------------
# Save results to JSON
# -------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Transcription complete. Results saved to {OUTPUT_JSON}")
