def update_visibility_matrix(visibilityArray, c_row, c_col, threshold, radius):
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
                if visArray[col, row] <= threshold:
                    visArray[col, row] = 0
                    changed = True
    return visArray, changed

def update_visibility_matrix_drone(visibilityArray, visibilityRange, c_row, c_col, flyRoute):
    temp_array = visibilityArray
    for i in range(len(visibilityRange)):
        temp_array, changed = update_visibility_matrix(temp_array, c_row, c_col, i+1, visibilityRange[i])
    flyRoute.append((c_row,c_col))
    return temp_array, changed, flyRoute