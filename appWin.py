import tkinter as tk
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageTk
from tkinter import ttk
from torch import autocast

# Create the main window
root = tk.Tk()
root.title("Image Generator")
root.geometry("1000x1000")

# Load the default model
default_model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(default_model_id, torch_dtype=torch.float32)

pipe = pipe.to("cuda")

# Function to generate the image
def generate_image():
    # Disable generate button and prompt entry during image generation
    generate_button.config(state=tk.DISABLED)
    prompt_entry.config(state=tk.DISABLED)

    prompt_text = prompt_entry.get("1.0", tk.END).strip()  # Get the text from the prompt entry
    image = pipe(prompt_text).images[0]
    generated_image_path = r"C:\Users\maxwe\Documents\Github repos\TextToImageAI\images\generated_image.jpg"  # Set the path for the generated image
    image.save(generated_image_path, format="JPEG")  # Save the generated image as a JPG

    # Open the generated image
    image = Image.open(generated_image_path)
    image = image.resize((768, 768))  # Resize the image
    image = ImageTk.PhotoImage(image)  # Convert the image to a PhotoImage object
    image_label.config(image=image)  # Update the image label
    image_label.image = image  # Store the image object in a label attribute

    # Enable generate button and prompt entry after image generation
    generate_button.config(state=tk.NORMAL)
    prompt_entry.config(state=tk.NORMAL)

# Function to generate image when Enter key is pressed in prompt entry
def generate_image_on_enter(event):
    generate_image()

# Function to update the model selection
def update_model(*args):
    selected_model = model_var.get()  # Get the selected model from the drop-down box
    global pipe  # Use global pipe variable for model update
    pipe = StableDiffusionPipeline.from_pretrained(selected_model, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

# Create input prompt entry
prompt_label = tk.Label(root, text="Enter Prompt:")
prompt_label.pack(pady=10)
prompt_entry = tk.Text(root, height=1, width=40)
prompt_entry.pack(pady=5)
prompt_entry.bind("<Return>", generate_image_on_enter)  # Bind Enter key to generate_image_on_enter function

# Create generate button
generate_button = tk.Button(root, text="Generate Image", command=generate_image)
generate_button.pack(pady=5)

# Create image label
image_label = tk.Label(root)
image_label.pack()

# Create model options
model_var = tk.StringVar()  # Create a StringVar to store the selected model
model_options = ttk.Combobox(root, textvariable=model_var, values=["runwayml/stable-diffusion-v1-5",
                                                                   "CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion",
                                                                   "dreamlike-art/dreamlike-photoreal-2.0","prompthero/openjourney"])  # Add your model options here
model_options.pack(pady=5)
model_options.bind("<<ComboboxSelected>>", update_model)  # Bind combobox selection event to update_model function

# Start the UI event loop
root.mainloop()

