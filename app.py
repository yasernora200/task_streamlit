import torch
from diffusers import DiffusionPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

prompt = input("Enter your prompt to generate an image: ")

image = pipe(prompt).images[0]

image.save("result.png")

print("Image generated and saved as result.png")