import librosa
import numpy as np

def load_audio(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def clean_audio(audio):
    # Apply noise reduction or normalization
    return audio  # Placeholder

def generate_spectrogram(audio, sr=22050):
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    return librosa.power_to_db(mel_spectrogram)