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

    def move_cell_adjacent(cell_a, cell_b):
        # Calculate the difference between the coordinates
        dx = cell_b[0] - cell_a[0]
        dy = cell_b[1] - cell_a[1]

        # Determine the direction
        direction = ""
        if dx < 0:
            direction += "N"
        elif dx > 0:
            direction += "S"
        if dy < 0:
            direction += "W"
        elif dy > 0:
            direction += "E"

        # Move cell B adjacent to cell A in the calculated direction
        new_x = cell_a[0]
        new_y = cell_a[1]

        if "N" in direction:
            new_x -= 1
        elif "S" in direction:
            new_x += 1

        if "W" in direction:
            new_y -= 1
        elif "E" in direction:
            new_y += 1

        new_c = (new_x, new_y)
        return new_c

    def are_adjacent(cell1, cell2):
        # Calculate the differences in x and y coordinates
        dx = abs(cell1[0] - cell2[0])
        dy = abs(cell1[1] - cell2[1])
        print(f"cell1 {cell1}, cell2 {cell2}")
        # Cells are adjacent if the difference is 1 in either direction
        # or both differences are 1 for diagonal adjacency
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1) or (dx == 1 and dy == 1)

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
    i = j = closest_index + 1
    start = closest_index + 1
    threshold = 10
    # Adjust cells before the new cell

    while not are_adjacent(original_route[i], original_route[i-1]) and original_route[i] != original_route[i-1]:
        original_route[i-1] = move_cell_adjacent(original_route[i],original_route[start-threshold])
        i -= 1
        if i == start - threshold:
            break

    while not are_adjacent(original_route[j], original_route[j+1]) and original_route[j] != original_route[j+1]:
        original_route[j+1] = move_cell_adjacent(original_route[j],original_route[start+threshold])
        j += 1
        if j == start + threshold:
            break

    def ensure_no_autoloop(lst):
        # Create an empty list to store the result
        result = []

        # Iterate through the original list
        for i in range(len(lst)):
            # If the result list is empty or the last element in result is different from the current one, append it
            if not result or lst[i] != result[-1]:
                result.append(lst[i])

        return result

    def ensure_continuity(route):

        # Create a new list to store the adjusted route
        adjusted_route = [route[0]]  # Start with the first cell

        # Iterate through the route and check if each pair of neighboring cells are adjacent
        for i in range(1, len(route)):
            current_cell = route[i - 1]
            next_cell = route[i]
            # If the cells are not adjacent, insert intermediate cells
            while not are_adjacent(current_cell, next_cell):
                # Calculate direction from current_cell to next_cell
                dx = next_cell[0] - current_cell[0]
                dy = next_cell[1] - current_cell[1]

                # Move in the direction of dx, dy towards next_cell
                if dx != 0:
                    current_cell = (current_cell[0] + (1 if dx > 0 else -1), current_cell[1])
                if dy != 0:
                    current_cell = (current_cell[0], current_cell[1] + (1 if dy > 0 else -1))

                # Insert the intermediate cell into the route
                adjusted_route.append(current_cell)

            # Once the cells are adjacent, add the next_cell to the adjusted route
            adjusted_route.append(next_cell)

        return adjusted_route

    return ensure_continuity(ensure_no_autoloop(original_route))