import cv2
import numpy as np
import os
from PIL import Image

# Define input and output directories
input_folder = "dt_logs/20250402_160829"  # Folder containing "Melted" and "Powder"
output_gif_folder = "output_gifs"

# Ensure output folder exists
if not os.path.exists(output_gif_folder):
    os.makedirs(output_gif_folder)


# Process images in each subfolder and create GIFs
for subfolder in ["optimization"]:
    input_subfolder = os.path.join(input_folder, subfolder)
    output_gif_path = os.path.join(output_gif_folder, f"{subfolder}.gif")

    images = []

    if os.path.exists(input_subfolder):
        for filename in sorted(os.listdir(input_subfolder)):  # Sort for frame order
            if filename.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(input_subfolder, filename)
                img = cv2.imread(img_path)

                # Convert to PIL format for GIF
                pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                images.append(pil_image)

    # Save as GIF if images exist
    if images:
        images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=300, loop=0)

