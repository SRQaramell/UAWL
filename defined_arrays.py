import numpy as np

'''Legend = np.array[[0], [255, 0, 0],'Burned out area, No search',
                 [1], [],'Very thick smoke, very poor visiblility',
                 [2], [],'Thick smoke/Very dense vgetation, poor visibility',
                 [3], [],'Smoke/dense vegetation, limited visibility',
                 [4], [],'Almost no smoke/medium vegetation,adequate visibility',
                 [5], [],'No smoke/very light vegetation, good visibility',
                 [6], [],'Plain area/open field, perfect visibility ',
                 [7], [255,255,255],'Water', 
                 [8], [], 'survivor/target'
'''

defined_rgb_values_NDVI = (np.array
([([0, 69, 0], 2), # very dense vegetation (RGB: [0, 69, 0], Number: 2)
([255, 255, 255], 7), # water (RGB: [255, 255, 255], Number: 7)
([222, 217, 156], 6), # sand/rock (RGB: [222, 217, 156], Number: 6)
 ([48, 110, 28], 3), # dense vegetation (RGB: [48, 110, 28], Number: 3)
 ([145, 191, 82], 5), # light vegetation (RGB: [145, 191, 82], Number: 5)
 ([163, 204, 89], 6),  # open field (RGB: [163, 204, 89], Number: 6)
 ([97, 150, 54], 4)], # medium vegetation (RGB: [97, 150, 54], Number: 4)
dtype=[('rgb', '3i4'), ('number', 'i4')]))

defined_rgb_values_FC = np.array([
    ([169, 160, 171], 1),  # dense-smoke (RGB: [169, 160, 171], Number: 1)
    ([103, 113, 132], 2),  # light smoke (RGB: [103, 113, 132], Number: 2)
    ([222, 217, 156], 6),  # ?)
    ([145, 191, 82], 6),   # ?)
    ([163, 204, 89], 6),   # ?)
    ([97, 150, 54], 6),    # ?)
], dtype=[('rgb', '3i4'), ('number', 'i4')])

defined_rgb_values_BAD = np.array([
    ([255, 0, 0], 0), #burned area
    ([0,0,0], 6),
],dtype=[('rgb', '3i4'), ('number', 'i4')])