import librosa
from librosa.effects import trim
import numpy as np
import os

def load_audio(file_path, sr=22050):
    audio = librosa.load(file_path, sr=sr)
    return audio

def clean_audio(audio):
    purified_audio = []
    purified_audio[0] = audio[0]/np.max(np.abs(audio[0]))
    purified_audio[0],purified_audio[2] = trim(audio)
    return purified_audio

def generate_spectrogram(audio, sr=22050):
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    return librosa.power_to_db(mel_spectrogram)

if __name__ == "__main__":
    path = 'data/SamHarrisMedi.mp3'
    audio = load_audio(path)
    print(audio[0].shape,audio[1])
    audio = clean_audio(audio)
    print(audio[0].shape,audio[1])