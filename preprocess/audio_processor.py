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
    cleansed_audio = {
        "time_series": None,      
        "true_length": None,  
        "offsets": None     
    }

    normalized_audio = audio / np.max(np.abs(audio))
    trimmed_audio, offsets = trim(normalized_audio)

    cleansed_audio["time_series"] = trimmed_audio
    cleansed_audio["true_length"] = audio.shape[0]
    cleansed_audio["offsets"] = offsets
    
    return cleansed_audio

def generate_spectrogram(audio, sr=22050):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=1024,hop_length=256)
    return librosa.power_to_db(mel_spectrogram)

if __name__ == "__main__":
    path = 'data/SamHarrisMedi.mp3'
    audio = load_audio(path)
    print(audio[0].shape)
    audio = clean_audio(audio[0])
    print(audio['time_series'].shape,audio['true_length'])
    
    mel_spec_db = generate_spectrogram(audio['time_series'])
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db, 
        sr=22050, 
        x_axis='time', 
        y_axis='mel', 
        cmap='magma'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.show()
