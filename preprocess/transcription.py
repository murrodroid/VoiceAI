from transformers import pipeline

def transcribe_audio(audio_path):
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    result = transcriber(audio_path)
    return result["text"]
