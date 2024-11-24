from PIL import Image

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
    height, width = array.shape

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