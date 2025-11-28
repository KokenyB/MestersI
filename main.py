from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/music-flamingo-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates."},
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/MF-Skills/resolve/main/assets/song_1.mp3"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
#print(decoded_outputs)
with open("./kimenet/output.txt", "w") as f:
    f.write(decoded_outputs)

#-------------------------------------------------------------------------------------------------

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

# Initialize the pipeline
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Qwen/Qwen-Image-Edit",
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="text_encoder/model*.safetensors"
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors"
        ),
    ],
    processor_config=ModelConfig(
        model_id="Qwen/Qwen-Image-Edit",
        origin_file_pattern="processor/"
    ),
)

# Load the Eigen-Banana-Qwen-Image-Edit LoRA
pipe.load_lora(pipe.dit, "eigen-ai-labs/eigen-banana-qwen-image-edit/eigen-banana-qwen-image-edit-fp16-lora.safetensors")

# Generate an initial image
prompt = decoded_outputs
input_image = pipe(
    prompt=prompt,
    seed=0,
    num_inference_steps=40,
    height=1328,
    width=1024
)
input_image.save("./kimenet/szovegtokep.jpg")

#------------------------------------------------------------------------------------------------

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_url = './kimenet/szovegtokep.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
#print(processor.decode(out[0], skip_special_tokens=True))

with open("./kimenet/output.txt", "w") as f:
    f.write(processor.decode(out[0], skip_special_tokens=True))