import os
import pyaudio
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
from google.oauth2.service_account import Credentials
from openai import OpenAI

# Path to your Google Cloud JSON key file
key_path = os.environ['GOOGLE_JSON'] # Path to your JSON File

# Google Speech-to-Text and Text-to-Speech credentials
credentials = Credentials.from_service_account_file(key_path)

# Google Speech-to-Text client
speech_client = speech.SpeechClient(credentials=credentials)

# Google Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_KEY'])

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

audio_stream = pyaudio.PyAudio().open(
    format=FORMAT, channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK)

print("Say something:")

# Capture audio
frames = [audio_stream.read(CHUNK) for _ in range(30)]  # 3 seconds
audio_data = b''.join(frames)

audio_stream.stop_stream()
audio_stream.close()

# Send audio to Google Speech-to-Text
audio = speech.RecognitionAudio(content=audio_data)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code='en-US')

response = speech_client.recognize(config=config, audio=audio)

if response.results and response.results[0].alternatives:
    transcription = response.results[0].alternatives[0].transcript
else:
    print("No transcription results found.")
    transcription = ""

# Send transcription to OpenAI
openai_response = client.completion.create(
  engine="text-davinci-003",
  prompt=transcription,
  max_tokens=50
).choices[0].text

# Convert OpenAI response to speech
synthesis_input = texttospeech.SynthesisInput(text=openai_response)
voice = texttospeech.VoiceSelectionParams(
    language_code='en-US',
    name='en-US-Wavenet-D')

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3)

tts_response = tts_client.synthesize_speech(
    input=synthesis_input, voice=voice,
    audio_config=audio_config)

# Save and play the response
with open('response.mp3', 'wb') as out:
    out.write(tts_response.audio_content)

os.system("mpg321 response.mp3")
