import numpy as np

def save_to_csv(array, filename):
    np.savetxt(filename, array, delimiter=",", fmt='%d')
    print(f"Array saved to {filename}")