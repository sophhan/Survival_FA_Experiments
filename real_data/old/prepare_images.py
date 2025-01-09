import os
from PIL import Image

def resize_and_save_images(input_folder, output_folder, size=(256, 256)):
    """
    Loads images from a folder with subfolders, resizes them, and saves them to a new folder structure 
    that mirrors the original structure.
    
    :param input_folder: Path to the input folder containing subfolders with images.
    :param output_folder: Path to the output folder where the resized images will be saved.
    :param size: Tuple specifying the target size (width, height) of the images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Walk through the input folder and process files
    for root, dirs, files in os.walk(input_folder):
        # Calculate the relative path of the current directory to the input folder
        relative_path = os.path.relpath(root, input_folder)
        # Create the corresponding directory in the output folder
        target_folder = os.path.join(output_folder, relative_path)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        for file in files:
            # Process only image files based on extensions
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(target_folder, file)

                try:
                    # Open the image, resize it, and save it to the output folder
                    with Image.open(input_file_path) as img:
                        img_resized = img.resize(size)
                        img_resized.save(output_file_path)
                        print(f"Saved resized image: {output_file_path}")
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")

# Define input and output folders
input_folder = '/opt/example-data/TCGA-SCNN/' # Path to the input folder
output_folder =  'data/'   # Path to the output folder

# Resize images to 256x256 pixels
resize_and_save_images(input_folder, output_folder, size=(512, 512))
