import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.set_page_config(page_title="Text to Image (SDXL)", layout="centered")

st.title("🎨 Text to Image Generator")
st.write("Write a prompt and AI will generate an image")

@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )

    pipe = pipe.to("cuda")
    return pipe

pipe = load_model()

prompt = st.text_area("Enter your prompt", "Astronaut in a jungle, cinematic lighting, ultra detailed")

steps = st.slider("Inference Steps", 10, 50, 30)
guidance = st.slider("Guidance Scale", 1.0, 10.0, 7.5)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]

    st.image(image, caption="Generated Image", use_container_width=True)

    image.save("result.png")
    st.success("Saved as result.png")