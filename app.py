import numpy as np
from PIL import Image
from image_to_number import process_image_to_number as img_to_nb
from defined_arrays import defined_rgb_values_NDVI as NDVI, defined_rgb_values_BAD as BAD, defined_rgb_values_FC as FC
from number_to_image import process_number_to_image as nb_to_img
from image_processing import image_processing as img_p



Img_dir = "Visual_data/NFVI_test.png"

#img_p(image_array,NDVI)
img_to_nb(Img_dir,NDVI)


#txt_filename = 'output.txt'  # The text file containing the array of numbers
#nb_to_img(txt_filename, NDVI)

