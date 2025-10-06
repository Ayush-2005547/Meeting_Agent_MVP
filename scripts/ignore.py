import whisper

model = whisper.load_model("base")
result = model.transcribe("sample_meeting.wav")
print(result["text"])
