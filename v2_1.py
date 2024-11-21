import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Processor.from_pretrained(model_name)

SAMPLE_RATE = 16000
DURATION = 15

sd.default.device = 6 # Esto varía mucho probado solo en ubuntu con el software pauvcontrol y el archivo device.py

def capture_audio_and_transcribe():
    print("Capturando audio del sistema...")
    try:
        audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()

        audio_tensor = torch.FloatTensor(audio_data.flatten())

        inputs = tokenizer(audio_tensor, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding="longest").input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis=-1)
        text = tokenizer.batch_decode(tokens)
        print("Transcripción:", text[0].lower())

    except Exception as e:
        print("Error al capturar/transcribir audio:", e)

while True:
    capture_audio_and_transcribe()


