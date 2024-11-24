from PIL import Image
import asyncio
import subprocess
from io import BytesIO
import os


async def show_image_async(image: Image.Image, duration=3):
    # Save the image to an in-memory buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Write the image to a temporary file
    temp_path = "temp_image.png"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(buffer.read())

    # Display the image using a subprocess
    viewer = None
    if os.name == "nt":  # Windows
        viewer = ["start", "/wait"]
    elif os.name == "posix":  # MacOS and Linux
        viewer = ["xdg-open"]

    process = subprocess.Popen(viewer + [temp_path], shell=True)
    await asyncio.sleep(duration)  # Wait asynchronously
    process.terminate()

    # Clean up temporary file
    os.remove(temp_path)

def generate_image_with_red_pixels(width, height, red_pixels):
    image = Image.new("RGB", (width, height), "black")
    pixels = image.load()

    for x, y in red_pixels:
        if 0 <= x < width and 0 <= y < height:  # Ensure the coordinates are within bounds
            pixels[x, y] = (255, 0, 0)  # Red color

    return image


def generate_image_from_array(array, red_pixels):
    # Define the color mapping for values 0 to 5
    color_map = {
        0: (0, 0, 0),  # Black
        1: (165, 0, 165),
        2: (0, 255, 0),  # Green
        3: (0, 0, 255),  # Blue
        4: (255, 255, 0),  # Yellow
        5: (255, 165, 0),  # Orange
    }

    # Get the dimensions of the array
    width, height = array.shape

    # Create an empty image
    image = Image.new("RGB", (width, height), "black")
    pixels = image.load()

    # Map array values to image pixels
    for y in range(height):
        for x in range(width):
            value = array[y, x]
            if value in color_map:
                pixels[x, y] = color_map[value]

    for x, y in red_pixels:
        if 0 <= x < width and 0 <= y < height:  # Ensure the coordinates are within bounds
            pixels[x, y] = (255, 0, 0)  # Red color

    return image

def create_gif(images, index):
    images[0].save(
        f"output{index}.gif",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=50,  # Duration per frame in milliseconds
        loop=0  # Loop forever
    )