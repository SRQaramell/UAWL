import numpy as np
from PIL import Image

def image_processing(img_dir, defined_rgb_values):
    image = Image.open(img_dir)
    image = image.convert("RGB")
    img_array = np.array(image)
    rgb_values = defined_rgb_values['rgb']

    height, width, _ = img_array.shape
    new_image_array = np.copy(img_array)

    for i in range(1, height - 1):
        for j in range(1, width - 1):

            block = img_array[i - 1:i + 2, j - 1:j + 2]


            if np.any(np.all(block == [255, 0, 0], axis=-1)):
                new_image_array[i, j] = rgb_values[0]
            elif defined_rgb_values['number'][0] == 0:
                new_image_array[i, j] = rgb_values[1]

            else:
                avg_rgb = np.mean(block, axis=(0, 1))
                distances = np.linalg.norm(rgb_values - avg_rgb, axis=1)
                closest_rgb = rgb_values[np.argmin(distances)]  #
                new_image_array[i, j] = closest_rgb

    new_image = Image.fromarray(new_image_array.astype(np.uint8))
    new_image.show()
    new_image.save("processed_image.png", format="PNG")