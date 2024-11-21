import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import os

model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Processor.from_pretrained(model_name)

SAMPLE_RATE = 16000
DURATION = 11
sd.default.device = 6

def get_next_file_name(base_name="prueba_", extension=".txt"):
    directorio = "results"
    counter = 1
    while True:
        file_name = f"{base_name}{counter:03d}{extension}"  
        if not os.path.exists(os.path.join(directorio, file_name)):  
            return file_name
        counter += 1


file_name = get_next_file_name()

def capture_audio_and_transcribe(file_name):
    directorio = "results"

    print("Preparándose para grabar en...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    
    try:
        audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        print("¡Grabando ahora!")
        sd.wait()
        
        audio_tensor = torch.FloatTensor(audio_data.flatten())

        inputs = tokenizer(audio_tensor, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding="longest").input_values
        logits = model(inputs).logits


        tokens = torch.argmax(logits, axis=-1)
        text = tokenizer.batch_decode(tokens)
        transcription = text[0].lower()

        print("Transcripción:", transcription)

        with open(os.path.join(directorio, file_name), "a") as file:
            file.write(transcription + "\n")

    except Exception as e:
        print("Error al capturar/transcribir audio:", e)


start_time = time.time()
MAX_DURATION = 12

while True:
    elapsed_time = time.time() - start_time
    if elapsed_time > MAX_DURATION:
        print("Tiempo máximo alcanzado. Finalizando el programa.")
        break
    capture_audio_and_transcribe(file_name)

