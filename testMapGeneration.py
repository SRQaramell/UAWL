import numpy as np
import random

def generateTestMap(width, height):
    return np.ones((height, width), dtype=int)


def add_irregular_patches(array, patch_number, coverage_percentage):
    height, width = array.shape
    total_cells = height * width
    target_cells = int(total_cells * (coverage_percentage / 100))
    added_cells = 0

    def add_irregular_patch(start_row, start_col):
        nonlocal added_cells
        patch_size = random.randint(5, 20)  # Define patch size range
        visited = set()
        stack = [(start_row, start_col)]

        while stack and len(visited) < patch_size:
            row, col = stack.pop()

            if (row, col) in visited or not (0 <= row < height and 0 <= col < width):
                continue

            if array[row, col] != patch_number:
                array[row, col] = patch_number
                added_cells += 1

            visited.add((row, col))

            # Add neighboring cells to the stack
            neighbors = [
                (row + 1, col), (row - 1, col),
                (row, col + 1), (row, col - 1),
                (row + 1, col + 1), (row - 1, col - 1),
                (row + 1, col - 1), (row - 1, col + 1),
            ]
            random.shuffle(neighbors)
            stack.extend(neighbors)

    while added_cells < target_cells:
        start_row = random.randint(0, height - 1)
        start_col = random.randint(0, width - 1)
        add_irregular_patch(start_row, start_col)

    return array