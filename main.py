
import numpy as np
from PIL import Image
from matplotlib import pyplot as PLT
from random import seed
from random import randint

# opens an image and creates an rgb array 
img = Image.open('wood256.png')
arr = np.array(img) 				

# display the graphic from the array of rgb values
# PLT.imshow(arr)
# PLT.show()

# gets the dimensions of the picture, z should be 3 for RGB values
x, y, z = arr.shape

# if there is a 360x360 png
# creates a 360x360x360x3 array
# assumes the texture is a square
solid = np.zeros((x,y,x,z))


print(solid[0,0,0].shape)
print(arr[0, 0].shape)

r = 0.8


# function to generate a random 3d structure from random pixels from the texture
def generate_random_solid(s):
	print("generating random solid texture\n")
	seed(s)
	for i in range(0, x):
		for j in range(0, x):
			for k in range(0, x):
				random_x = randint(0, x-1)
				random_y = randint(0, x-1)
				solid[i, j, k] = arr[random_x][random_y]
		
		# mini percentage calculator
		if i % 10 == 0:
			print(i/x)
	print("finished generating random solid texture\n")

# uncomment to call func to generate random solid
# WARNING: Takes a long time!!!
# generate_random_solid(1)



# function to find the nearest neighbors on the texture when passed in a 
# neighborhood and a point from the 3D solid
def neighborhoods(pixels_x, pixels_y, pixels_z):
	# assume neighborhoods are 5x5

	# Input: Neighborhoods for vectorized neighborhoods for each voxel
	# in the slices orthogonal to the x, y, and z axis

	# Output: Exemplar neighborhoods on the original texture
	
	min_x_point = (0,0)
	min_y_point = (0,0)
	min_z_point = (0,0)

	min_x_value = np.inf
	min_y_value = np.inf
	min_z_value = np.inf

	for i in range(2, x-2):
		for j in range(2, x-2):
			cur_point = (i, j)
			cur_array = [arr[a,b] for a in range(i-2, i+3) for b in range(j-2, j+3)]

			# Assertions
			assert len(cur_array) == len(pixels_x)
			assert len(cur_array) == len(pixels_y)
			assert len(cur_array) == len(pixels_z)

			# Calculate X
			total_x = 0
			total_y = 0
			total_z = 0

			# calculate L2
			for k in range(len(cur_array)):
				total_x += np.linalg.norm(cur_array[k] - pixels_x[k])
				total_y += np.linalg.norm(cur_array[k] - pixels_y[k])
				total_z += np.linalg.norm(cur_array[k] - pixels_z[k])

			if total_x < min_x_value:
				min_x_point = (i, j)
				min_x_value = total_x
			
			if total_y < min_y_value:
				min_y_point = (i, j)
				min_y_value = total_y
			
			if total_z < min_z_value:
				min_z_point = (i, j)
				min_z_value = total_z

	return [min_x_point, min_y_point, min_z_point]


# neighborhood testing
def test_neighborhoods():
	test_arr_x = [[0, 0, 0] for i in range(25)]
	test_arr_y = [[0, 0, 0] for i in range(25)]
	test_arr_z = [[0, 0, 0] for i in range(25)]

	for i in range(25):
		random_x = randint(0, x-1)
		random_y = randint(0, x-1)
		test_arr_x[i] = arr[random_x][random_y]
		random_x = randint(0, x-1)
		random_y = randint(0, x-1)
		test_arr_y[i] = arr[random_x][random_y]
		random_x = randint(0, x-1)
		random_y = randint(0, x-1)
		test_arr_z[i] = arr[random_x][random_y]

	res = neighborhoods(test_arr_x, test_arr_y, test_arr_z)
	print(res)

test_neighborhoods()

			 























