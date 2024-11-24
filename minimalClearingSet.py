def find_minimal_cells_to_map_clear(matrix, visibilityRange, clear_function):

    finalRoute = set()
    matCop = matrix.copy()

    for i in range(len(visibilityRange) -1, -1, -1):
        tempRoute, matCop =  find_minimal_cells_to_clear(matCop,visibilityRange[i], i+1,clear_function)
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