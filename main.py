import os
import streamlit as st
import numpy as np
import speech_recognition as sr
import whisper
import torch
from transformers import pipeline
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

# Initialize the recognizer and the Whisper model
recognizer = sr.Recognizer()

# Load Whisper model
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
import os
import numpy as np
import whisper
import torch
import wave
import speech_recognition as sr
from vad import EnergyVAD

# Initialize SpeechRecognition and Microphone
recognizer = sr.Recognizer()

# Initialize EnergyVAD
vad = EnergyVAD(
    sample_rate=16000,
    frame_length=20,  # in milliseconds
    frame_shift=20,   # in milliseconds
    energy_threshold=0.1,  # you may need to adjust this value
    pre_emphasis=0.95  # default values
)

def record_and_transcribe():
    with sr.Microphone(sample_rate=16000) as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        
        print("Listening for speech...")

        recording = False
        frames = []

        while True:
            audio = recognizer.listen(source)
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
import google.generativeai as genai
def generate_response(transcription):
# Set your API key directly
    api_key = "AIzaSyBA2rSQh0CNpWq5s_AOjoPzFXqxlLWv67E"

    # Configure the generative AI with the API key
    genai.configure(api_key=api_key)

    # Define the model and the query
    model = genai.GenerativeModel('gemini-1.5-flash')
    query = transcription

    # Generate content using the model
    response = model.generate_content(query, generation_config=genai.types.GenerationConfig(
        stop_sequences=["."],
        max_output_tokens=60,
        temperature=0.6,
    ))

    # Print the generated response
    return response.text
import asyncio
import edge_tts

# Function to synthesize speech with Edge TTS
async def synthesize_speech(text, voice="en-US-AriaNeural", rate="+0%", pitch="+0Hz", output_file="output.wav"):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    with open(output_file, 'wb') as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
st.title("Speech-to-Text-to-Speech App")
if st.button("Start Recording"):
    transcription = record_and_transcribe()
    st.write("Transcription:", transcription)
    response = generate_response(transcription)
    st.write("Generated Response:", response)
    st.subheader("Text-to-Speech Settings")

if st.button("Generate Speech"):
    voice = "en-US-AriaNeural" if input("Select Voice: \n 1. Female \n 2. Male")==1 else "en-US-GuyNeural" # Example: en-US-AriaNeural (Female), en-US-GuyNeural (Male)
    rate = str(input("Enter Rate + for Higher - For lower")+"%")  # Speech speed (e.g., "-20%" for slower, "+20%" for faster)
    pitch = str(input("Enter Pitch + for Higher - For lower")+"Hz")  # Pitch (e.g., "+100Hz" for higher, "-100Hz" for lower)
    output_file = "output.wav"

    synthesize_speech(response, voice, rate, pitch, output_file)

    st.write("Speech synthesized and saved")
