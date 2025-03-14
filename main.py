import os
import librosa
import librosa.display
from preprocess.audio_processor import load_audio,clean_audio,generate_spectrogram
from preprocess.transcription import transcribe_audio_whisper
from training.trainer import fine_tune_model
import matplotlib.pyplot as plt

def test():
    path = 'data/SamHarrisMedi.mp3'
    audio,sr = load_audio(path)
    transcript = transcribe_audio_whisper(path)
    
    print(transcript)
    
    # trimmed_audio,offsets = clean_audio(audio)

    # mel_spec_db = generate_spectrogram(trimmed_audio)
    # print(mel_spec_db)

def main(model_name):
    # Input audio file
    
    # Preprocess the audio
    processed_audio = preprocess_audio(input_audio)
    
    # Generate transcript
    transcript = transcribe_audio(input_audio)
    
    # Fine-tune the voice model
    trained_model = fine_tune_model(processed_audio, transcript)
    
    # Save the trained model
    model_path = f"checkpoints/{model_name}.pth"
    
    # trained_model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    test()