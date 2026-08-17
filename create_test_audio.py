import pyttsx3
import os

print("[SETUP] Creating a valid audio file for testing...")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
audio_path = os.path.join("data", "lecture_sample.wav")

engine = pyttsx3.init()
text = "Welcome to Exam Hub. Today we will focus on Data Warehousing and Computer Networks for the 2026 syllabus."
engine.save_to_file(text, audio_path)
engine.runAndWait()

print(f"[SETUP] Success! Audio saved to: {audio_path}")