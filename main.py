
import numpy as np
from PIL import Image
from matplotlib import pyplot as PLT

# opens an image and creates an rgb array 
img = Image.open('wood360.png')
exemplar = np.array(img) 				

# display the graphic from the array of rgb values
PLT.imshow(exemplar)
PLT.show()

# gets the dimensions of the picture, z should be 3 for RGB values
exemplar_x, exemplar_y, exemplar_z = exemplar.shape
print(exemplar_x)

# assuming that the solid texture is a cube (so exemplar_x = exemplar_y)
sol_dim = exemplar_x
channel_dim = exemplar_z

# 3D solid texture
solid = np.zeros(shape=(sol_dim, sol_dim, sol_dim, channel_dim))

# 4D array for matching exemplar neighborhood of each voxel's slices (value for each index is the center of the exemplar neighborhoods)
matching = np.zeros(shape=(sol_dim, sol_dim, sol_dim, 3))

# 4D array for weights during optimization phase
matching = np.zeros(shape=(sol_dim, sol_dim, sol_dim, 3))

# initialize solid texture by getting random values from the exemplar and set matches 
for i in range(sol_dim):
    for j in range(sol_dim):
        for k in range(sol_dim):
            x = np.random.rand() * sol_dim
            y = np.random.rand() * sol_dim
            for c in range(channel_dim):
                solid[i][j][k][c] = exemplar[x][y][c]
            # set all matching to the random neighborhoods for now
            for s in range(3):
                matching[i][j][k][l] = (x, y)

# OPTIMIZATION PHASE
neighborhood_size = 16

'''
Returns the vectorized 2D array representing the neighborhood in the exemplar given the index 
of the center of the desired neighborhood. If we extend beyond the exemplar, pad with 
zeros
'''
def get_neighborhood_ex(center):
    x, y = center
    width = neighborhood_size / 2
    x_start = max(0, x - width)
    x_end = min(exemplar_x, x + width)
    y_start = max(0, y - width)
    y_end = min(exemplar_y, y + width)
    result = np.zeros(neighborhood_size**2 * channel_dim)
    x_indices = zip(range(0, neighborhood_size), range(x_start, x_end))
    y_indices = zip(range(0, neighborhood_size), range(y_start, y_end))
    for i, k in x_indices:
        for j, l in y_indices:
            for c in range(channel_dim):
                result[k * l + c] = exemplar[i][j][c]
    return result




'''
Returns the vectorized 2D array representing the sliced neighborhood in the solid given the index 
of the center of the desired neighborhood and the axis of the neighborhood as 
a string. If we extend beyond the solid, pad with zeros
'''
def get_neighborhood_sol(center, axis="x"):
    x, y, z = center
    width = neighborhood_size / 2
    result = np.zeros(neighborhood_size**2 * channel_dim)
    if axis == "z":
        x_start = max(0, x - width)
        x_end = min(sol_dim, x + width)
        y_start = max(0, y - width)
        y_end = min(sol_dim, y + width)
        x_indices = zip(range(0, neighborhood_size), range(x_start, x_end))
        y_indices = zip(range(0, neighborhood_size), range(y_start, y_end))
        for i, k in x_indices:
            for j, l in y_indices:
                for c in range(channel_dim):
                    result[k * l + c] = sol_dim[i][j][z][c]
    elif axis == "y":
        x_start = max(0, x - width)
        x_end = min(sol_dim, x + width)
        z_start = max(0, z - width)
        z_end = min(sol_dim, z + width)
        x_indices = zip(range(0, neighborhood_size), range(x_start, x_end))
        z_indices = zip(range(0, neighborhood_size), range(z_start, z_end))
        for i, k in x_indices:
            for j, l in z_indices:
                for c in range(channel_dim):
                    result[k * l + c] = sol_dim[i][y][j][c]
    else: 
        y_start = max(0, y - width)
        y_end = min(sol_dim, y + width)
        z_start = max(0, z - width)
        z_end = min(sol_dim, z + width)
        y_indices = zip(range(0, neighborhood_size), range(y_start, y_end))
        z_indices = zip(range(0, neighborhood_size), range(z_start, z_end))
        for i, k in y_indices:
            for j, l in z_indices:
                for c in range(channel_dim):
                    result[k * l + c] = sol_dim[x][i][j][c]
    return result


# calculate weights for each voxel for each voxel's three slices
for i in range(sol_dim):
    for j in range(sol_dim):
        for k in range(sol_dim):
            for s in ["x", "y", "z"]:
                pass


