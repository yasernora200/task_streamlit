import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.set_page_config(page_title="Anime Generator", layout="centered")

st.title("🎨 Anime Image Generator (Z-Anime)")

@st.cache_resource
def load_model():
    model_id = "SeeSee21/Z-Anime"

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )

    # use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    return pipe

pipe = load_model()

prompt = st.text_area(
    "Enter your prompt:",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
)

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]

    st.image(image, caption="Generated Anime Image", use_column_width=True)