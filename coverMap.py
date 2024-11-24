import numpy as np
from utils import save_to_csv
from visibilityMatrix import update_visibility_matrix_drone
import testMapGeneration
from visibilityAndRouteVisualisation import generate_image_from_array
from routeGeneration import find_route_to_closest_non_black
import math
from routeOptimization import insert_and_move_closest

visibilityRange = [25,23,20,18,15]
flyRoute = []
priorityMap = []

def optimize_map(visibilityarray, startPos):

    orgArray = visibilityarray.copy()
    flyRoute.clear()
    clearedArray, needsOptimizing, partRoute, targetLocation = clear_map(visibilityarray.copy(), startPos)
    while needsOptimizing:
        newRoute = insert_and_move_closest(partRoute, targetLocation)
        partClearedArray = clear_map_given_path(visibilityarray.copy(), newRoute)
        clearedArray, needsOptimizing, partRoute, targetLocation = clear_map(partClearedArray, newRoute[-1])
        image = generate_image_from_array(clearedArray,flyRoute)
        image.show()

    finalArray = clear_map_given_path(orgArray, flyRoute)
    image = generate_image_from_array(finalArray, flyRoute)
    image.show()

def clear_map(visibilityArray, startPos):
    tempArray = visibilityArray
    pos = startPos
    cutRoute = []
    problematicLocation = []
    needOptimizing = False
    flyRoute.clear()
    while np.any(tempArray > 0):  # Continue while there are non-black pixels
        # Update the visibility matrix and record the route
        tempArray, changed = update_visibility_matrix_drone(tempArray, visibilityRange, pos[1], pos[0], flyRoute)

        # Find the next route if changes occurred
        if changed:
            newRoute = find_route_to_closest_non_black(tempArray, pos)
            print(len(newRoute))
            if len(newRoute) > 2*visibilityRange[0] and needOptimizing == False:
                needOptimizing = True
                problematicLocation = newRoute[-1]
                cutRoute = flyRoute.copy()

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

    return tempArray, needOptimizing, cutRoute, problematicLocation

def clear_map_given_path(matrix, path):
    mat = matrix.copy()
    route = []
    for posit in path:
        mat, changed = update_visibility_matrix_drone(mat,visibilityRange,posit[0], posit[1], route)
    return mat

baseMap = testMapGeneration.generateTestMap(100,100)
mapWith2 = testMapGeneration.add_irregular_patches(baseMap,2,40)
mapWith3 = testMapGeneration.add_irregular_patches(mapWith2,3,23)
mapWith4 = testMapGeneration.add_irregular_patches(mapWith3,4,11)
mapWith5 = testMapGeneration.add_irregular_patches(mapWith4,5,5)
finalMap = mapWith5.copy()
save_to_csv(mapWith5, "test.csv")
optimize_map(finalMap, (0,50))