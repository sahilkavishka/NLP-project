import os
import soundfile as sf
from transformers import pipeline

class AudioTranscriber:
    """
    A professional module to handle Audio to Text transcription using Whisper AI.
    Designed to be easily integrated into a Django Backend.
    """
    def __init__(self, model_name="openai/whisper-tiny"):
        print(f"[AudioTranscriber] Initializing model: {model_name}...")
        # Load the model once when the class is instantiated
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name)
        print("[AudioTranscriber] Model loaded successfully.")

    def transcribe(self, file_path):
        """
        Reads a .wav file directly and returns the transcribed text.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] Audio file not found at: {file_path}")
        
        print(f"[AudioTranscriber] Reading audio file: {file_path}")
        
        # Read using soundfile to bypass ffmpeg dependencies
        audio_data, sample_rate = sf.read(file_path)
        
        # Convert stereo to mono if necessary
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        print("[AudioTranscriber] Processing audio data. Please wait...")
        
        # Execute transcription
        result = self.transcriber({"array": audio_data, "sampling_rate": sample_rate})
        transcript = result.get("text", "").strip()
        
        print("[AudioTranscriber] Transcription completed.")
        return transcript

# ==========================================
# Module Testing Block
# (This only runs if this file is executed directly, not when imported in Django)
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    # Point to the audio file created in step 1
    test_file = "../data/lecture_sample.wav"
    
    try:
        transcriber = AudioTranscriber()
        text_output = transcriber.transcribe(test_file)
        
        print("\n--- FINAL TRANSCRIPT ---")
        print(text_output)
        print("------------------------\n")
    except Exception as e:
        print(f"Error during execution: {e}")