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
                    changed = True
    return visArray, changed

def update_visibility_matrix_drone(visibilityArray, visibilityRange, c_col, c_row, flyRoute):
    temp_array = visibilityArray
    for i in range(len(visibilityRange)):
        temp_array, changed = update_visibility_matrix(temp_array, c_col, c_row, i+1, visibilityRange[i])
    flyRoute.append((c_col,c_row))
    return temp_array, changed, flyRoute