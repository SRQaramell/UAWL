import math

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
