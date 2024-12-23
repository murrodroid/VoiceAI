import os
from preprocess.audio_processor import preprocess_audio
from preprocess.transcription import transcribe_audio
from training.trainer import fine_tune_model

def main(model_name):
    # Input audio file
    input_audio = "data/raw/input_voice.wav"
    
    # Preprocess the audio
    processed_audio = preprocess_audio(input_audio)
    
    # Generate transcript
    transcript = transcribe_audio(input_audio)
    
    # Fine-tune the voice model
    trained_model = fine_tune_model(processed_audio, transcript)
    
    # Save the trained model
    model_path = f"checkpoints/{model_name}.pth"
    trained_model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main('far')