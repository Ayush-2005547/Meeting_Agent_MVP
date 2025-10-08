import whisper
import soundfile as sf

# Load audio without ffmpeg
audio, sr = sf.read("../data/sample_meeting.wav")
model = whisper.load_model("base")

result = model.transcribe(audio, fp16=False, samplerate=sr)
print(result["text"])