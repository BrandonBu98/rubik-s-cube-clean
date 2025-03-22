import os
import cv2
import pillow_heif

# Define input and output folder
input_folder = "test/image_tile"  # Replace with your folder containing JPG/HEIC images
output_folder = "train/image_sticker"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic"):  # Only process HEIC files
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

        # Read HEIC image and convert to PNG
        heif_file = pillow_heif.open_heif(image_path)
        image = heif_file.to_pillow()  # Convert HEIC to PIL image
        image.save(output_path, "PNG")  # Save as PNG

        print(f"Converted: {filename} -> {output_path}")

print("Batch HEIC to PNG conversion completed!")
