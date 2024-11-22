import numpy as np
import random
from queue import PriorityQueue
from PIL import Image

visibilityRange = [30,27,24,21,18]
flyRoute = []

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


def save_to_csv(array, filename):
    np.savetxt(filename, array, delimiter=",", fmt='%d')
    print(f"Array saved to {filename}")

def update_visibility_matrix(visibilityArray, c_col, c_row, threshold, radius):
    height, width = visibilityArray.shape
    changed = False
    for row in range(c_row - radius, c_row + radius + 1):
        for col in range(c_col - radius, c_col + radius + 1):
            if (
                    0 <= row < height and
                    0 <= col < width and
                    (row - c_row) ** 2 + (col - c_col) ** 2 <= radius ** 2
            ):
                if visibilityArray[row, col] <= threshold:
                    visibilityArray[row, col] = 0
                    changed = True
    return visibilityArray, changed

def update_visibility_matrix_drone(visibilityArray, visibilityRange, c_col, c_row):
    temp_array = visibilityArray
    for i in range(len(visibilityRange)):
        temp_array, changed = update_visibility_matrix(temp_array, c_col, c_row, i+1, visibilityRange[i])
    return temp_array, changed

def column_flyby(visibilityArray, column):
    height, width = visibilityArray.shape
    temp_array = visibilityArray
    if column <= width:
        for i in range(height):
            temp_array = update_visibility_matrix_drone(temp_array, visibilityRange, column, i)
            flyRoute.append((column, i))
    return temp_array


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

def find_route_to_closest_non_black(array, start_pos):
    """
    Finds the closest non-black pixel from the given position and creates a route to it.

    Parameters:
        array (numpy.ndarray): 2D array where 0 represents black pixels.
        start_pos (tuple): Starting position (row, col).

    Returns:
        list: List of positions [(row1, col1), (row2, col2), ...] forming the route.
              If no non-black pixel is found, returns an empty list.
    """
    rows, cols = array.shape
    visited = set()
    queue = PriorityQueue()

    # Add the starting position to the priority queue
    queue.put((0, start_pos))  # (distance, position)
    visited.add(start_pos)

    # Directions for moving in 8 directions (N, S, E, W, NE, NW, SE, SW)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Dictionary to track the parent of each cell (for reconstructing the route)
    parent = {start_pos: None}

    # BFS with priority (closest distance first)
    while not queue.empty():
        dist, (row, col) = queue.get()

        # If the pixel is non-black, reconstruct the route
        if array[row, col] != 0:
            route = []
            current = (row, col)
            while current is not None:
                route.append(current)
                current = parent[current]
            return route[::-1]  # Reverse to get the path from start to target

        # Explore neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)

            if (0 <= new_row < rows and 0 <= new_col < cols and
                new_pos not in visited):
                visited.add(new_pos)
                parent[new_pos] = (row, col)  # Track the parent for backtracking
                queue.put((dist + 1, new_pos))

    return []  # Return empty list if no non-black pixel is found

def clear_map(visibilityArray, startPos):
    tempArray = visibilityArray
    pos = startPos
    while np.any(tempArray > 0):  # Continue while there are non-black pixels
        # Update the visibility matrix and record the route
        tempArray, changed = update_visibility_matrix_drone(tempArray, visibilityRange, pos[1], pos[0])
        flyRoute.append(pos)

        # Find the next route if changes occurred
        if changed:
            newRoute = find_route_to_closest_non_black(tempArray, pos)
            print(newRoute)

            if not newRoute:  # If no route is found, exit the loop
                break

            # Move to the next position in the route
            old_pos = pos
            pos = newRoute.pop(0)
            if pos == old_pos and len(newRoute) > 0:
                pos = newRoute.pop(0)
        else:
            # If no changes were made, break to avoid infinite loops
            break

    return tempArray

baseMap = generateTestMap(100,100)
mapWith2 = add_irregular_patches(baseMap,2,40)
mapWith3 = add_irregular_patches(mapWith2,3,23)
mapWith4 = add_irregular_patches(mapWith3,4,11)
mapWith5 = add_irregular_patches(mapWith4,5,5)
save_to_csv(mapWith5, "test.csv")
mapFlyby = clear_map(mapWith5, (50,0))
save_to_csv(mapFlyby, "testFly.csv")
image = generate_image_from_array(mapFlyby,flyRoute)
#singleClear, changed = update_visibility_matrix_drone(mapWith5, visibilityRange, 150,150)
#print(find_route_to_closest_non_black(singleClear, (150,140)))
#image = generate_image_from_array(singleClear, flyRoute)
image.show()