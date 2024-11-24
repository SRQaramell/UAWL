import numpy as np
from utils import save_to_csv
from visibilityMatrix import update_visibility_matrix_drone
import testMapGeneration
from visibilityAndRouteVisualisation import generate_image_from_array, create_gif
from routeGeneration import find_route_to_closest_non_black
from routeOptimization import insert_and_move_closest, insert_and_adjust_route

visibilityRange = [15,13,11,9,7]
images = []

def optimize_map(visibilityarray, startPos):
    ind = 0
    orgArray = visibilityarray.copy()
    clearedArray, needsOptimizing, partRoute, targetLocation = clear_map(visibilityarray.copy(), startPos, [], ind)
    ind += 1
    while needsOptimizing:
        newRoute = insert_and_adjust_route(partRoute, targetLocation)
        print(f"New route len {len(newRoute)}, part route len {len(partRoute)}")
        partClearedArray = clear_map_given_path(visibilityarray.copy(), newRoute)
        image = generate_image_from_array(partClearedArray, newRoute)
        image.show()
        clearedArray, needsOptimizing, partRoute, targetLocation = clear_map(partClearedArray, newRoute[-1], newRoute, ind)
        ind += 1

    finalArray = clear_map_given_path(orgArray, partRoute)
    image = generate_image_from_array(finalArray, partRoute)
    image.show()

def clear_map(visibilityArray, startPos, startRoute, ind):
    tempArray = visibilityArray
    pos = startPos
    prevRoute = startRoute
    cutRoute = []
    problematicLocation = []
    needOptimizing = False
    images = []
    while np.any(tempArray > 0):  # Continue while there are non-black pixels
        # Update the visibility matrix and record the route
        tempArray, changed, prevRoute = update_visibility_matrix_drone(tempArray, visibilityRange, pos[1], pos[0], prevRoute)
        images.append(generate_image_from_array(tempArray, prevRoute))

        # Find the next route if changes occurred
        if changed:
            newRoute = find_route_to_closest_non_black(tempArray, pos)
            if len(newRoute) > 2*visibilityRange[0] and needOptimizing == False:
                needOptimizing = True
                problematicLocation = (newRoute[-1][1], newRoute[-1][0])
                cutRoute = prevRoute.copy()

            if not newRoute:  # If no route is found, exit the loop
                break

        # Move to the next position in the route
        old_pos = pos
        pos = newRoute.pop(0)
        if pos == old_pos and len(newRoute) > 0:
            pos = newRoute.pop(0)
        #else:
            # If no changes were made, break to avoid infinite loops
            # break
    create_gif(images, ind)
    ind += 1
    if needOptimizing == True:
        return tempArray, needOptimizing, cutRoute, problematicLocation
    return tempArray, needOptimizing, prevRoute, problematicLocation

def clear_map_given_path(matrix, path):
    mat = matrix.copy()
    route = []
    for posit in path:
        mat, changed, route = update_visibility_matrix_drone(mat,visibilityRange,posit[0], posit[1], route)
    return mat

baseMap = testMapGeneration.generateTestMap(100,100)
mapWith2 = testMapGeneration.add_irregular_patches(baseMap,2,40)
mapWith3 = testMapGeneration.add_irregular_patches(mapWith2,3,23)
mapWith4 = testMapGeneration.add_irregular_patches(mapWith3,4,11)
mapWith5 = testMapGeneration.add_irregular_patches(mapWith4,5,5)
finalMap = mapWith5.copy()
save_to_csv(mapWith5, "test.csv")
optimize_map(finalMap, (0,50))