from transformers import pipeline
from torch.cuda import is_available

def transcribe_audio(audio_path: str, model_name: str = "openai/whisper-tiny") -> tuple[list[str], list[tuple[float, float]]]:
    """
    Transcribe the given audio file and return a tuple of:
      1) text: a list of strings
      2) timestamps: a list of (start_time, end_time) in seconds for each string

    :param audio_path: Path to the audio file (e.g., .wav, .mp3).
    :param model_name: Name of the Hugging Face ASR model to use (default: openai/whisper-tiny).
                      Some popular choices: openai/whisper-small, openai/whisper-medium, openai/whisper-large-v2, etc.
    :return: (text, timestamps)
    """

    device = 0 if is_available() else -1

    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=device
    )

    output = transcriber(audio_path, return_timestamps=True)

    text = []
    timestamps = []

    if "chunks" in output:
        for chunk in output["chunks"]:
            text_chunk = chunk["text"].strip()
            start_time, end_time = chunk["timestamp"]

            text.append(text_chunk)
            timestamps.append((start_time, end_time))

    else:
        text = [output["text"]]
        timestamps = [(0.0, 0.0)]
    
    return text, timestamps
    
def transcribe_audio_whisper(audio):
    model = 'openai/whisper-tiny'
    transcription = transcribe_audio(audio,model)
    return transcription


def transcribe_audio_old(audio_path):
    device = "cuda" if is_available() else "cpu"
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device, return_timestamps=True)
    chunks = transcriber(audio_path)['chunks']

    text = []
    timestamps = []
    for chunk in chunks:
        text.append(chunk['text'])
        timestamps.append(chunk['timestamp'])

    return text,timestamps