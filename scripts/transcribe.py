import os
import json
import whisper
import whisper
import soundfile as sf

# Load audio without ffmpeg
audio, sr = sf.read("../data/sample_meeting.wav")
model = whisper.load_model("base")

result = model.transcribe(audio, fp16=False, samplerate=sr)
print(result["text"])
# -------------------------
# Paths
# -------------------------
AUDIO_FILE = "../data/sample_meeting.wav"      # single WAV file
OUTPUT_JSON = "../data/sample_meetings.json"

# -------------------------
# Load Whisper model
# -------------------------
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded.")

# -------------------------
# Transcribe the file
# -------------------------
print(f"Transcribing: {AUDIO_FILE} ...")
result = model.transcribe(AUDIO_FILE)
text = result["text"].strip()

results = [{
    "meeting_id": os.path.basename(AUDIO_FILE),
    "generated": text,
    "reference": ""  # can be filled later
}]

print(f"Done:\n{text}\n")

# -------------------------
# Save results to JSON
# -------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Transcription complete. Results saved to {OUTPUT_JSON}")
