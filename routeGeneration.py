from queue import PriorityQueue

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