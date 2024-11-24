import math
import numpy as np

def insert_and_move_closest(route, target):

    def euclidean_distance(a, b):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def is_adjacent(a, b):
        """Check if two points are adjacent in a grid."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    # Find the index of the tuple closest to the target
    closest_idx = min(range(len(route)), key=lambda i: euclidean_distance(route[i], target))
    closest_point = route[closest_idx]

    # Determine the direction to move closer to the target
    dx = target[0] - closest_point[0]
    dy = target[1] - closest_point[1]
    move = (int(dx / abs(dx)) if dx != 0 else 0, int(dy / abs(dy)) if dy != 0 else 0)
    new_point = (closest_point[0] + move[0], closest_point[1] + move[1])

    # Insert the new cell into the route
    new_route = route[:closest_idx] + [new_point] + route[closest_idx:]

    # Fix the route to maintain adjacency
    for i in range(len(new_route) - 1):  # Adjust forward and backward
        if not is_adjacent(new_route[i], new_route[i + 1]):
            dx = new_route[i + 1][0] - new_route[i][0]
            dy = new_route[i + 1][1] - new_route[i][1]
            new_route[i + 1] = (new_route[i][0] + (dx // abs(dx) if dx != 0 else 0),
                                new_route[i][1] + (dy // abs(dy) if dy != 0 else 0))

    return new_route


def insert_and_adjust_route(original_rou, target_cell):

    original_route = original_rou.copy()

    def distance(cell1, cell2):
        """Calculate Manhattan distance between two cells."""
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

    # Find the index of the closest cell in the route
    distances = [distance(cell, target_cell) for cell in original_route]
    closest_index = np.argmin(distances)

    # Insert new cell halfway between closest cell and target cell
    closest_cell = original_route[closest_index]
    new_cell = (
        (closest_cell[0] + target_cell[0]) // 2,
        (closest_cell[1] + target_cell[1]) // 2,
    )
    original_route.insert(closest_index + 1, new_cell)

    # Ensure continuity by adjusting nearby cells
    i = closest_index + 1
    route_length = len(original_route)

    # Adjust cells before the new cell
    for j in range(i - 1, 0, -1):
        if distance(original_route[j], original_route[j - 1]) > 1:
            if j >= 15:  # If fixing breaks 15+ cells, insert new ones instead
                diff = np.array(original_route[j]) - np.array(original_route[j - 1])
                step = diff // abs(diff).max()
                intermediate = tuple(np.array(original_route[j - 1]) + step)
                original_route.insert(j, intermediate)
            else:
                original_route[j - 1] = (
                    original_route[j][0] - np.sign(original_route[j][0] - original_route[j - 1][0]),
                    original_route[j][1] - np.sign(original_route[j][1] - original_route[j - 1][1]),
                )

    # Adjust cells after the new cell
    for j in range(i + 1, route_length):
        if distance(original_route[j], original_route[j - 1]) > 1:
            if route_length - j >= 15:  # If fixing breaks 15+ cells, insert new ones instead
                diff = np.array(original_route[j]) - np.array(original_route[j - 1])
                step = diff // abs(diff).max()
                intermediate = tuple(np.array(original_route[j - 1]) + step)
                original_route.insert(j, intermediate)
            else:
                original_route[j] = (
                    original_route[j - 1][0] + np.sign(original_route[j][0] - original_route[j - 1][0]),
                    original_route[j - 1][1] + np.sign(original_route[j][1] - original_route[j - 1][1]),
                )

    return original_route
