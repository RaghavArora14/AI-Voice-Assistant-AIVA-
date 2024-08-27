import os
import numpy as np
import whisper
import torch
import wave
import speech_recognition as sr
from transformers import pipeline
from vad import EnergyVAD
vad = EnergyVAD(
    sample_rate=16000,
    frame_length=20,  # in milliseconds
    frame_shift=20,   # in milliseconds
    energy_threshold=0.5,  # you may need to adjust this value
    pre_emphasis=0.95  # default values
)
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    def record_and_transcribe(self):
        with sr.Microphone(sample_rate=16000) as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            
            print("Listening for speech...")

            recording = False
            frames = []

            while True:
                audio = self.recognizer.listen(source)
                audio_np = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                voice_activity = vad(audio_np)

                # Check if any frame is detected as speech
                if any(voice_activity):
                    if not recording:
                        print("Voice detected, starting recording...")
                        recording = True

                    frames.append(audio.get_raw_data())
                else:
                    if recording:
                        print("Silence detected, stopping recording...")
                        recording = False
                        break
            audio_filename="recording.wav"

            # Save the recorded audio to a file
            with wave.open(audio_filename, 'wb') as wf:
                wf.setnchannels(1)  # mono audio
                wf.setsampwidth(2)  # sample width in bytes (2 bytes for int16)
                wf.setframerate(16000)  # frame rate (sample rate)
                wf.writeframes(b''.join(frames))
            print(f"Audio recorded and saved as {audio_filename}")
            
            # Transcribe the recorded audio
            print("Transcribing audio with Whisper...")
            result = whisper_pipeline(audio_filename)
            print("Transcription:", result['text'])
            return result['text']

