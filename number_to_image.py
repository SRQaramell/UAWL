import numpy as np
import matplotlib.pyplot as plt

def process_number_to_image(txt_filename, predefined_table):
    with open(txt_filename, 'r') as file:
        lines = file.readlines()

    number_array = []
    for line in lines:
        number_array.append(list(map(int, line.strip().split(','))))

    number_array = np.array(number_array)

    rgb_dict = {num: rgb for rgb, num in predefined_table}

    height, width = number_array.shape
    image_array = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            number = number_array[i, j]
            image_array[i, j] = rgb_dict.get(number, [0, 0, 0])

    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    plt.savefig('output_image.png', bbox_inches='tight')
    print("Image saved as output_image.png")