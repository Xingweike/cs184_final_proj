/*
 
 Installing opencv instructions
 https://www.codementor.io/@ohasanli/opencv-on-xcode-142qxx3sl8

 */

#include <highgui.hpp>
#include <core.hpp>
#include <iostream>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace cv;
using namespace std;

// Variables
const int neighbor_size = 5;
const int neighbor_half = 2; // floor nieghbor_size /  2
const int neighbor_size_2 = neighbor_size * neighbor_size;
const int tex_size = 256;

int array_R[tex_size][tex_size] = {};
int array_G[tex_size][tex_size] = {};
int array_B[tex_size][tex_size] = {};

int solid[tex_size][tex_size][tex_size][3] = {};

// Functions
void generate_random_solid();
void neighborhoods (int pix_x[], int pix_y[], int pix_z[], int size);

int main() {
    
    // Read image into buffer
    Mat image = imread("/Users/tomliu/Desktop/cs184/final_proj_xcode/184final/wood256.png");
    Vec3b buf;
    
    if(!image.data){
       cout <<  "Could not open or find the image" << std::endl ;
       return -1;
    }

    // Write image into the three arrays
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            buf = image.at<Vec3b>(i,j);
            array_B[i][j] = buf[0];
            array_G[i][j] = buf[1];
            array_R[i][j] = buf[2];
        }
    }
    
    // generate solid of random RGB pixels from original texture
    // Does not take long at all!
    generate_random_solid();

    // Show image!
    // imshow("Image",image);
    // waitKey(0);
}

void generate_random_solid() {
    
    // initialize seed
    srand (1234);
    
    // get random pixels from the original array
    for (int x = 0; x < tex_size; x += 1) {
        for (int y = 0; y < tex_size; y += 1) {
            for (int z = 0; z < tex_size; z += 1) {
                int ran_x = rand() % tex_size;  // random number from 0 to 255
                int ran_y = rand() % tex_size;
                solid[x][y][z][0] = array_R[ran_x][ran_y];
                solid[x][y][z][1] = array_G[ran_x][ran_y];
                solid[x][y][z][2] = array_B[ran_x][ran_y];
            }
        }
        if (x % 10 == 0) {
            cout << "Progress: " << round(x * 100 / tex_size) << "%" << endl;
        }
    }
    cout << "Generating random solid complete!" << endl;
}

// Input: (neighbor_size * neighbor_size) amount of pixels for each slice orthogonal to the x, y, and z axis
// Output: [3][neighbor_size * neighbor_size] array representing best neighborhoods in the original templar
void neighborhoods (int pix_x[neighbor_size_2][3], int pix_y[neighbor_size_2][3], int pix_z[neighbor_size_2][3]) {
    int slice_x[neighbor_size_2][3] = {};
    int slice_y[neighbor_size_2][3] = {};
    int slice_z[neighbor_size_2][3] = {};
    
    int temp[neighbor_size_2][3] = {};
    
    int min_x_point = 999;
    int min_y_point = 999;
    int min_z_point = 999;
    
    // iterate over each pixel in original texture
    for (int x = neighbor_half; x < tex_size - neighbor_half; x += 1) {
        for (int y = neighbor_half; y < tex_size - neighbor_half; y += 1) {
            
            // get the neighbor_size_2 number of points surrounding (i, j)
            int i = 0;
            for (int temp_x = x - neighbor_half; temp_x < x + neighbor_half; temp_x += 1) {
                for (int temp_y = y - neighbor_half; temp_y < y + neighbor_half; temp_y += 1) {
                    temp[i][0] = array_R[temp_x][temp_y];
                    temp[i][1] = array_G[temp_x][temp_y];
                    temp[i][2] = array_B[temp_x][temp_y];
                    i += 1;
                }
            }
            
            int total_x = 0;
            int total_y = 0;
            int total_z = 0;
            int cur_x[neighbor_size_2][3] = {};
            int cur_y[neighbor_size_2][3] = {};
            int cur_z[neighbor_size_2][3] = {};
            
            // subtract the arrays
            for (int i = 0; i < neighbor_size_2; i += 1) {
                
                cur_x[i][0] = temp[i][0] - pix_x[i][0];
                cur_x[i][1] = temp[i][1] - pix_x[i][1];
                cur_x[i][2] = temp[i][2] - pix_x[i][2];
                
                cur_y[i][0] = temp[i][0] - pix_y[i][0];
                cur_y[i][1] = temp[i][1] - pix_y[i][1];
                cur_y[i][2] = temp[i][2] - pix_y[i][2];
                
                cur_z[i][0] = temp[i][0] - pix_z[i][0];
                cur_z[i][1] = temp[i][1] - pix_z[i][1];
                cur_z[i][2] = temp[i][2] - pix_z[i][2];
                
                
                
            }
            
            // calculate L2 norm
            // cv::norm (InputArray src1, int normType=NORM_L2, InputArray mask=noArray())
        }
    }
}
