import numpy as np
from PIL import Image
from image_to_number import process_image_to_number as img_to_nb
from defined_arrays import defined_rgb_values_NDVI as NDVI, defined_rgb_values_BAD as BAD, defined_rgb_values_FC as FC
from number_to_image import process_number_to_image as nb_to_img
from image_processing import image_processing as img_p



Img_dir1 = "Visual_data/2017-07-17-00_00_2017-07-17-23_59_Sentinel-2_L2A_Burned_Area_Detection.png"
Img_dir2 = "Visual_data/2017-07-17-00_00_2017-07-17-23_59_Sentinel-2_L2A_Normalized_Difference_Vegetation_Index_(NDVI).png"
Img_dir3 = "Visual_data/2017-07-17-00_00_2017-07-17-23_59_Sentinel-2_L2A_False_Color.png"
img_p(Img_dir3,FC)
#img_to_nb(Img_dir,NDVI)


#txt_filename = 'output.txt'  # The text file containing the array of numbers
#nb_to_img(txt_filename, NDVI)

