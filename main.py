import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, BlipProcessor, BlipForConditionalGeneration
from diffusers import AutoPipelineForText2Image
import requests
import warnings
from PIL import Image

# 1. Beállítások meghatározása (gpu ellenőrzése)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

# 2. Modell és Processzor betöltése (Whisper)
print(f"Modell betöltése: {model_id} a(z) {device} eszközre...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    dtype=dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# 3. Pipeline létrehozása átírásra
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
    return_timestamps=True,
    language='english'
)

# 4. Bemeneti hangfájl megadása
audio_file_path = "sybau.mp3"  #<-------- Hangfájl elérési útja
transcribed_text = ""

try:
    print(f"\nIndul a beszédátírás a(z) '{audio_file_path}' fájlról...")
    result = pipe(audio_file_path)
    transcribed_text = result["text"]
    # 5. Eredmény kiírása a konzolra
    print("\n--- ÁTÍRÁS EREDMÉNYE ---")
    print(transcribed_text)
    print("------------------------")
    
except Exception as e:
    print(f"Hiba történt a fájl feldolgozása közben: {e}")

    


# Ellenőrizzük, hogy van-e átírt szöveg a hiba elkerülése érdekében
if transcribed_text:
    print("\nIndul az összefoglalás Ollama-n keresztül...")

    # 1. Prompt meghatározása
    ollama_prompt = (
        f"Give me a short summary in 20 to 60 words max"
        f"a következő szöveget: '{transcribed_text}'"
    )

    # 2. API kérés összeállítása
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:latest",  # Itt adod meg a helyi Ollama modell nevét
        "prompt": ollama_prompt,
        "stream": False,     # Egyszerű válasz kérése, nem streamelve
        "options": {
            "num_ctx": 4096  # Bővíti a kontextus ablakot (ha szükséges)
        }
    }
    
    summarized_text = transcribed_text 

    try:
        # 3. Kérés elküldése
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status() # Hiba dobása, ha a kérés sikertelen

        # 4. Válasz feldolgozása
        ollama_data = response.json()
        
        # A lényeges válasz általában a 'response' mezőben van
        summarized_text = ollama_data.get('response', '').strip()

        print("\n--- ÖSSZEFOGLALÁS EREDMÉNYE (OLLAMA) ---")
        print(summarized_text)
        print("------------------------------------------")

    except requests.exceptions.RequestException as e:
        print(f"Hiba az Ollama API elérésekor: {e}")
        summarized_text = transcribed_text 
        
else:
    summarized_text = "Nincs feldolgozható szöveg."
    print("Nincs átírt szöveg, az összefoglalás és a képgenerálás kihagyva.")




if summarized_text and summarized_text != "Nincs feldolgozható szöveg.":

    # Képgenerálás a szövegből
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    prompt = summarized_text
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    image.show()

    # Kép magyarázat
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = image.convert('RGB')

    print("------------------------------------------")
    text = "A képen látható:"
    inputs = processor(raw_image, text, return_tensors="pt")


    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    print("------------------------------------------")