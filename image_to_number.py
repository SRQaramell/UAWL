import numpy as np
from PIL import Image

def save_array_to_txt(new_number_array, filename):
    with open(filename, 'w') as f:
        for row in new_number_array:
            f.write(','.join(map(str, row)) + '\n')


def process_image_to_number(img_dir,defined_rgb_values):
    image = Image.open(img_dir)
    image = image.convert("RGB")
    img_array = np.array(image)
    rgb_values = defined_rgb_values['rgb']
    rgb_number = defined_rgb_values['number']

    height, width, _ = img_array.shape

    new_height = height // 3
    new_width = width // 3

    new_number_array = np.zeros((new_height, new_width), dtype=int)

    for i in range(new_height):
        for j in range(new_width):
            block = img_array[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]

            if np.any(np.all(block == [255, 0, 0], axis=-1)):
                new_number_array[i, j] = rgb_number[0]
            elif rgb_number[0] == 0:
                new_number_array[i, j] = rgb_number[1]
            else:
                avg_rgb = np.mean(block, axis=(0, 1))

                distances = np.linalg.norm(rgb_values - avg_rgb, axis=1)

                closest_index = np.argmin(distances)

                closest_number = rgb_number[closest_index]

                new_number_array[i, j] = closest_number

    save_array_to_txt(new_number_array,'output.txt')
    print(new_number_array)