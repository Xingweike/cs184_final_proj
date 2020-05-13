import numpy as np
from PIL import Image
from matplotlib import pyplot as PLT
from random import seed
from random import randint

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
			print("percent: " + str(i * 100 / x))
	print("saving solid to file\n")
	np.save('random_arr.npy', solid) 



# function to find the nearest neighbors on the texture when passed in a 
# neighborhood and a point from the 3D solid
def neighborhoods(pixels_x, pixels_y, pixels_z):
	# assume neighborhoods are 5x5

	# Input: Neighborhoods for vectorized neighborhoods for each voxel
	# in the slices orthogonal to the x, y, and z axis

	# Output: neighborhoods in the original templar
	
	min_x_point = [[] for i in range(25)]
	min_y_point = [[] for i in range(25)]
	min_z_point = [[] for i in range(25)]

	min_x_value = np.inf
	min_y_value = np.inf
	min_z_value = np.inf
	

	# iterate over each pixel in the texture
	for i in range(2, x-2):
		for j in range(2, x-2):

			# array of pixels in the neighborhood
			cur_point = (i, j)
			cur_array = [arr[a,b] for a in range(i-2, i+3) for b in range(j-2, j+3)]

			# Assertions for testing
			# assert len(cur_array) == len(pixels_x)
			# assert len(cur_array) == len(pixels_y)
			# assert len(cur_array) == len(pixels_z)

			# Calculate X, Y, Z
			total_x = 0
			total_y = 0
			total_z = 0

			# calculate L2
			for k in range(len(cur_array)):
				total_x += np.linalg.norm(cur_array[k] - pixels_x[k])
				total_y += np.linalg.norm(cur_array[k] - pixels_y[k])
				total_z += np.linalg.norm(cur_array[k] - pixels_z[k])

			if total_x < min_x_value:
				min_x_point = cur_array
				min_x_value = total_x
			
			if total_y < min_y_value:
				min_y_point = cur_array
				min_y_value = total_y
			
			if total_z < min_z_value:
				min_z_point = cur_array
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
	
	for cur in res:
		print(cur)
		print("\n\n")


# import image
img = Image.open('wood256.png')
arr = np.array(img) 
exemplar = arr				

# display the graphic from the array of rgb values
# PLT.imshow(arr)
# PLT.show()

# gets the dimensions of the picture, z should be 3 for RGB values
exemplar_x, exemplar_y, exemplar_z = exemplar.shape
x, y, z = arr.shape

# assuming that the solid texture is a cube (so exemplar_x = exemplar_y)
sol_dim = exemplar_x
channel_dim = exemplar_z

# 3D solid texture, assumes the texture is a square
solid = np.zeros(shape=(sol_dim, sol_dim, sol_dim, channel_dim))

# 4D array for matching exemplar neighborhood of each voxel's slices (value for each index is the center of the exemplar neighborhoods)
matching = np.zeros(shape=(sol_dim, sol_dim, sol_dim, 3))

# 4D array for weights during optimization phase
matching = np.zeros(shape=(sol_dim, sol_dim, sol_dim, 3))



# exponent to make the result less vulnerable to outliers
r = 0.8

# neighbothood size
neighborhood_size = 5


# uncomment to call func to generate random solid given a random seed
# WARNING: Takes a long time!!!
# generate_random_solid(1234)

# load solid from file instead
solid = np.load('random_arr.npy')
print("loaded random solid texture from file!")

# test the function to find the nearest neighbors on the texture
test_neighborhoods()	 






















