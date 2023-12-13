import os
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from torch import autocast

# Directory where models will be saved
MODEL_DIR = 'C:/Users/maxwe/Documents/Github repos/TextToImageAI/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to save model
def save_model(model, model_name):
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pt')
    try:
        torch.save(model, model_path)
    except Exception as e:
        st.write(f"Error saving model: {e}")

# Function to load model
def load_model(model_id, model_name):
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pt')
    if os.path.isfile(model_path):
        try:
            return torch.load(model_path)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    else:
        model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        save_model(model, model_name)
        return model

# List of tuples, where each tuple contains the model id and the filename
model_options = [("runwayml/stable-diffusion-v1-5", "runwayml_stable_diffusion_v1_5"),
                 ("CompVis/stable-diffusion-v1-4", "CompVis_stable_diffusion_v1_4"),
                 ("hakurei/waifu-diffusion", "hakurei_waifu_diffusion"),
                 ("dreamlike-art/dreamlike-photoreal-2.0", "dreamlike_art_dreamlike_photoreal_2_0"),
                 ("prompthero/openjourney", "prompthero_openjourney")]

# Load the default model
default_model_id, default_model_name = model_options[0]
pipe = load_model(default_model_id, default_model_name)
pipe = pipe.to("cuda")

def generate_image(prompt_text):
    image = pipe(prompt_text).images[0]
    image.save("generated_image.png")  # Save the generated image
    image = Image.open("generated_image.png")  # Open the generated image
    image = image.resize((768, 768))  # Resize the image
    return image

def update_model(selected_model_id, selected_model_name):
    global pipe  # Use global pipe variable for model update
    pipe = load_model(selected_model_id, selected_model_name)
    pipe = pipe.to("cuda")

st.title('Image Generator')

# Model names for the selectbox
model_names = [name for id, name in model_options]
selected_model_name = st.selectbox('Select Model', model_names)
# Get the selected model id
selected_model_id = [id for id, name in model_options if name == selected_model_name][0]

update_model(selected_model_id, selected_model_name)

prompt_text = st.text_input('Enter Prompt')

if st.button('Generate Image'):
    img = generate_image(prompt_text)
    st.image(img)
