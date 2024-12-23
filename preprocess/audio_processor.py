import librosa
import librosa
from librosa.effects import trim
import numpy as np
import os
import matplotlib.pyplot as plt

def load_audio(file_path):
    audio,sr = librosa.load(file_path)
    return audio,sr

def clean_audio(audio):
    normalized_audio = audio / np.max(np.abs(audio))
    trimmed_audio, offsets = trim(normalized_audio)
    return trimmed_audio, offsets

def generate_spectrogram(audio, sr=22050):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=1024,hop_length=256)
    return librosa.power_to_db(mel_spectrogram)

if __name__ == "__main__":
    pass
