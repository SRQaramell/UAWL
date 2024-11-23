from collections import deque

import numpy as np
import random
from queue import PriorityQueue
from PIL import Image
import heapq
import math

visibilityRange = [30,28,25,22,19]
flyRoute = []
priorityMap = []

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
    visArray = visibilityArray.copy()
    height, width = visArray.shape
    changed = False
    for row in range(c_row - radius, c_row + radius + 1):
        for col in range(c_col - radius, c_col + radius + 1):
            if (
                    0 <= row < height and
                    0 <= col < width and
                    (row - c_row) ** 2 + (col - c_col) ** 2 <= radius ** 2
            ):
                if visArray[row, col] <= threshold:
                    visArray[row, col] = 0
                    #priorityMap[row, col] = 0
                    changed = True
                #elif priorityMap[row, col] != 0:
                #    priorityMap[row, col] = priorityMap[row, col] + 1
    return visArray, changed

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
            #newRoute = find_route_to_closest_non_black_priority(tempArray, pos, visibilityRange[0])
            newRoute = bfs_route(tempArray, pos, visibilityRange[0])

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


def find_minimal_cells_to_map_clear(matrix):

    finalRoute = set()
    matCop = matrix.copy()

    for i in range(len(visibilityRange) -1, -1, -1):
        tempRoute, matCop =  find_minimal_cells_to_clear(matCop,visibilityRange[i], i+1,update_visibility_matrix)
        finalRoute.update(tempRoute)
    return finalRoute


def find_minimal_cells_to_clear(matrix, radius, target, clear_function):

    copyMatrix = matrix.copy()
    rows, cols = matrix.shape
    target_cells = set((i, j) for i in range(rows) for j in range(cols) if copyMatrix[i, j] == target)
    clearing_centers = []

    while target_cells:
        # Find the best cell to clear maximum targets
        max_cleared = 0
        best_center = None

        for i, j in target_cells:
            # Simulate the effect of clearing from this cell
            cleared = {
                (x, y)
                for x in range(max(0, i - radius), min(rows, i + radius + 1))
                for y in range(max(0, j - radius), min(cols, j + radius + 1))
                if (x - i) ** 2 + (y - j) ** 2 <= radius ** 2 and (x, y) in target_cells
            }
            if len(cleared) > max_cleared:
                max_cleared = len(cleared)
                best_center = (i, j)

        # Use the best center to clear targets
        if best_center:
            clearing_centers.append(best_center)
            copyMatrix, changed = clear_function(copyMatrix, best_center[1], best_center[0], 5, radius)
            target_cells = set((i, j) for i, j in target_cells if copyMatrix[i, j] == target)
        else:
            break  # No more target cells can be cleared
        if not changed:
            print("No changes made, breaking loop.")
            break

    return clearing_centers, copyMatrix

def dijkstra_2d_solution_matrix(matrix, source, goal, must_visit_nodes):
    rows, cols = len(matrix), len(matrix[0])
    must_visit_index = {node: idx for idx, node in enumerate(must_visit_nodes)}
    m = len(must_visit_nodes)

    # Initialize the distance dictionary
    distance = {}
    predecessors = {}  # To track the path
    for i in range(rows):
        for j in range(cols):
            distance[(i, j)] = {}
            predecessors[(i, j)] = {}

    # Priority queue
    Q = []

    # Push the source node with an empty visited set and zero cost
    initial_visited_set = frozenset()
    heapq.heappush(Q, (0, source[0], source[1], initial_visited_set))  # (cost, row, col, visited_set)
    distance[(source[0], source[1])][initial_visited_set] = 0
    predecessors[(source[0], source[1])][initial_visited_set] = None

    # Directions for neighbors (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while Q:
        # Get the node with the lowest cost
        cost, u_row, u_col, visited_set = heapq.heappop(Q)

        # Skip if the cost is not the minimum stored
        if cost != distance[(u_row, u_col)].get(visited_set, math.inf):
            continue

        # Check neighbors
        for dr, dc in directions:
            v_row, v_col = u_row + dr, u_col + dc
            if 0 <= v_row < rows and 0 <= v_col < cols:  # Ensure within bounds
                new_visited_set = visited_set

                # Update the visited set if the neighbor is a must-visit node
                if (v_row, v_col) in must_visit_index:
                    vid = must_visit_index[(v_row, v_col)]
                    new_visited_set = visited_set | frozenset([vid])

                # Calculate the new cost (all edges cost 1)
                new_cost = cost + 1

                # Update if the new cost is smaller
                if new_cost < distance[(v_row, v_col)].get(new_visited_set, math.inf):
                    distance[(v_row, v_col)][new_visited_set] = new_cost
                    heapq.heappush(Q, (new_cost, v_row, v_col, new_visited_set))
                    predecessors[(v_row, v_col)][new_visited_set] = (u_row, u_col, visited_set)

    # Find the minimal distance where all must-visit nodes are visited
    full_visited_set = frozenset(range(m))
    min_distance = math.inf
    best_final_state = None

    for visited_set, dist in distance[goal].items():
        if full_visited_set.issubset(visited_set):
            if dist < min_distance:
                min_distance = dist
                best_final_state = visited_set

    if min_distance == math.inf:
        return -1, []  # No path found

    # Reconstruct the path
    path = []
    current = (goal[0], goal[1])
    visited_set = best_final_state

    while current is not None:
        path.append((current[0], current[1]))
        prev = predecessors[current][visited_set]
        if prev is not None:
            current, visited_set = (prev[0], prev[1]), prev[2]
        else:
            current = None

    path.reverse()
    return min_distance, path

def clear_map_given_path(matrix, path):
    mat = matrix.copy()
    for posit in path:
        mat, changed = update_visibility_matrix_drone(mat,visibilityRange,posit[0], posit[1])
    return mat


# Define the 8 possible directions for movement (vertical, horizontal, and diagonal)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def is_valid_move(array, x, y):
    """Check if the position (x, y) is within the bounds of the array."""
    return 0 <= x < array.shape[0] and 0 <= y < array.shape[1]


def heuristic(a, b):
    """Calculate the Manhattan or diagonal distance heuristic between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(array, start, goals):
    """A* algorithm to find the shortest path from the start to the closest goal."""
    if not goals:  # If no goals, return path to start (self)
        goals = [start]

    # Priority queue to store nodes to explore: (cost, (x, y), path)
    open_list = []
    heapq.heappush(open_list, (0, start, []))

    # Set of visited nodes to avoid cycles
    visited = set()
    visited.add(start)

    while open_list:
        cost, current, path = heapq.heappop(open_list)
        x, y = current

        # If we reached one of the goals, return the path
        if current in goals:
            return path + [current]

        # Explore neighbors
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if is_valid_move(array, nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                new_cost = cost + 1  # Assuming uniform cost for each move
                new_path = path + [current]
                heapq.heappush(open_list, (new_cost + heuristic((nx, ny), goals[0]), (nx, ny), new_path))

    return []  # If no path found


def find_route(array, start, goal_list=[]):
    """Find the route from the starting cell to the closest cell in goal_list."""
    # Find the closest goal
    if goal_list:
        closest_goal = min(goal_list, key=lambda goal: heuristic(start, goal))
        goal_list.remove(closest_goal)
        return a_star(array, start, [closest_goal])
    else:
        return a_star(array, start, [])


def create_path(array, start, goal_list):
    """Create a path from the starting point, visiting the closest point from the goal list each time."""
    full_path = []
    current_position = start
    remaining_goals = goal_list.copy()

    while remaining_goals:
        # Find the closest goal from the current position
        closest_goal = min(remaining_goals, key=lambda goal: heuristic(current_position, goal))

        # Find the path from the current position to the closest goal
        path_to_goal = find_route(array, current_position, [closest_goal])

        if path_to_goal:
            # Add the path to the full path
            full_path.extend(path_to_goal[1:])  # Skip current_position (since it's already in path_to_goal)
            current_position = closest_goal
            remaining_goals.remove(closest_goal)
        else:
            break  # No valid path to the goal, break the loop

    # After visiting all goals, return to the start
    if current_position != start:
        return_to_start_path = find_route(array, current_position, [start])
        if return_to_start_path:
            full_path.extend(return_to_start_path[1:])  # Skip current_position (since it's already in return_to_start_path)

    return full_path


def find_route_to_closest_non_black_priority(array, start_pos, radius):
    rows, cols = array.shape
    visited = set()
    queue = PriorityQueue()

    # Add the starting position to the priority queue
    queue.put((0, start_pos, None))  # (distance, position, direction)
    visited.add(start_pos)

    # Directions for moving in 8 directions (N, S, E, W, NE, NW, SE, SW)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Dictionary to track the parent of each cell (for reconstructing the route)
    parent = {start_pos: None}
    direction_parent = {start_pos: None}  # Tracks direction of movement for each cell

    # Function to calculate Manhattan distance to the closest edge
    def edge_priority(row, col):
        return min(row, rows - 1 - row, col, cols - 1 - col)

    # Function to calculate the "priority" of a direction to continue
    def direction_priority(last_direction, direction):
        if last_direction is None:
            return 0  # Starting cell has no direction, equal priority
        # Same direction gets a higher priority
        if last_direction == direction:
            return 0  # Prioritize continuing in the same direction
        return 1  # Other directions are less prioritized

    # BFS with priority (closest distance first)
    while not queue.empty():
        dist, (row, col), last_direction = queue.get()

        # If the pixel is non-black and outside the radius, prioritize by edge proximity and number
        if array[row, col] != 0 and (abs(row - start_pos[0]) > radius or abs(col - start_pos[1]) > radius):
            # Reconstruct the path to the current cell
            route = []
            current = (row, col)
            while current is not None:
                route.append(current)
                current = parent[current]
            return route[::-1]  # Return the reversed path from start to goal

        # Explore neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)

            if (0 <= new_row < rows and 0 <= new_col < cols and
                    new_pos not in visited):
                visited.add(new_pos)
                parent[new_pos] = (row, col)  # Track the parent for backtracking
                direction_parent[new_pos] = (dr, dc)  # Track the direction of movement
                priority = dist + 1 + direction_priority(last_direction, (dr, dc))
                queue.put((dist + 1, new_pos, (dr, dc)))  # Add the new position with direction

    return []  # Return empty list if no non-black pixel is found

# Check if a cell is closer to the edge
def is_edge(cell, matrix_shape):
    row, col = cell
    rows, cols = matrix_shape
    return row == 0 or row == rows - 1 or col == 0 or col == cols - 1

# Calculate the Manhattan distance to the nearest edge
def distance_to_edge(cell, matrix_shape):
    row, col = cell
    rows, cols = matrix_shape
    return min(row, rows - 1 - row, col, cols - 1 - col)

def bfs_route(matrix, start, radius):
    rows, cols = matrix.shape
    # Step 1: Identify all cells just outside the radius (one unit away)
    outside_radius_cells = []
    for r in range(rows):
        for c in range(cols):
            if abs(r - start[0]) > radius or abs(c - start[1]) > radius:
                if abs(r - start[0]) == radius or abs(c - start[1]) == radius:
                    outside_radius_cells.append((r, c))

    # If no valid cells outside the radius, expand the search to the next layer
    if not outside_radius_cells:
        for r in range(rows):
            for c in range(cols):
                if abs(r - start[0]) > radius or abs(c - start[1]) > radius:
                    outside_radius_cells.append((r, c))

    # Step 2: Get all cells with values greater than 0
    valid_cells = [cell for cell in outside_radius_cells if matrix[cell] > 0]

    if not valid_cells:
        return None, None  # No valid cells found

    # Step 3: Get the cells with the lowest values
    min_value = np.min([matrix[cell] for cell in valid_cells])  # Ensure we're checking values > 0
    lowest_cells = [cell for cell in valid_cells if matrix[cell] == min_value]

    # Step 4: Find the cell among those that is closest to the edge
    closest_to_edge_cell = min(lowest_cells, key=lambda cell: distance_to_edge(cell, matrix.shape))
    print(closest_to_edge_cell)

    # Step 5: Calculate the route from start to the chosen destination
    route = []
    queue = deque([start])
    visited = np.zeros_like(matrix, dtype=bool)
    visited[start] = True
    parent_map = {start: None}

    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while queue:
        x, y = queue.popleft()
        if (x, y) == closest_to_edge_cell:
            break
        for dx, dy in direction:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                visited[nx, ny] = True
                parent_map[(nx, ny)] = (x, y)
                queue.append((nx, ny))

    # Backtrack to find the full route
    cell = closest_to_edge_cell
    while cell != start:
        route.append(cell)
        cell = parent_map[cell]

    route.append(start)
    route.reverse()

    return route

baseMap = generateTestMap(100,100)
#priorityMap = baseMap
mapWith2 = add_irregular_patches(baseMap,2,40)
mapWith3 = add_irregular_patches(mapWith2,3,23)
mapWith4 = add_irregular_patches(mapWith3,4,11)
mapWith5 = add_irregular_patches(mapWith4,5,5)
finalMap = mapWith5.copy()
save_to_csv(mapWith5, "test.csv")
#to_clear = find_minimal_cells_to_map_clear(mapWith5)
#print(to_clear)
#dist, path = dijkstra_2d_solution_matrix(mapWith5,(15,0),(15,0),to_clear)
#path = create_path(finalMap, (15,0),to_clear)
#print(path)
#cleared = clear_map_given_path(finalMap, path)
#image = generate_image_from_array(cleared, path)
#image.show()

mapFlyby = clear_map(finalMap, (50,0))
save_to_csv(mapFlyby, "testFly.csv")
image = generate_image_from_array(mapFlyby,flyRoute)
#singleClear, changed = update_visibility_matrix_drone(mapWith5, visibilityRange, 150,150)
#print(find_route_to_closest_non_black(singleClear, (150,140)))
#image = generate_image_from_array(singleClear, flyRoute)
image.show()