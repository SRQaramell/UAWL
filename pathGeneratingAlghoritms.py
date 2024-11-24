import heapq
import math
from collections import deque

import numpy as np


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