
import numpy as np
from PIL import Image
from matplotlib import pyplot as PLT


# opens an image and creates an rgb array 
img = Image.open('wood360.png')
arr = np.array(img) 				

# display the graphic from the array of rgb values
PLT.imshow(arr)
PLT.show()

# gets the dimensions of the picture, z should be 3 for RGB values
x, y, z = arr.shape
print(x)

# 3d Array of zeros
np.zeros(shape=(n,n,n))