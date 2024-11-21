import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
from pydub import AudioSegment
import io

model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Processor.from_pretrained(model_name)

r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
    print("Speak Anything :")
    
    while True:
        try:
            audio = r.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            clip = AudioSegment.from_file(data)
            
            x = torch.FloatTensor(clip.get_array_of_samples())
            
            inputs = tokenizer(x,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="longest").input_values
            
            logits = model(inputs).logits
            tokens = torch.argmax(logits, axis=-1)
            text = tokenizer.batch_decode(tokens)
            
            print("You said:", str(text).lower())
            
        except Exception as e:
            
            print("Error:", e)

            
            