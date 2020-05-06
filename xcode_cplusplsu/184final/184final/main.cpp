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

#include <algorithm>
#include <iterator>

#include <chrono>


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
void test_neighbor();

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
//    generate_random_solid();

    // Show image!
    // imshow("Image",image);
    // waitKey(0);
    
    test_neighbor();
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
            
            // Trying roi to get points surrounding each point on the texture
//            Rect region_of_interest = Rect(bl_x, bl_y, tl_x, tl_y);
//            Mat image_roi = image(region_of_interest);
            
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
            for (int i = 0; i < neighbor_size_2; i += 1) {
                total_x += sqrt(
                                pow(temp[i][0] - pix_x[i][0], 2) +
                                pow(temp[i][1] - pix_x[i][1], 2) +
                                pow(temp[i][2] - pix_x[i][2], 2)
                                );
                
                total_y += sqrt(
                                pow(temp[i][0] - pix_y[i][0], 2) +
                                pow(temp[i][1] - pix_y[i][1], 2) +
                                pow(temp[i][2] - pix_y[i][2], 2)
                                );
                
                total_z += sqrt(
                                pow(temp[i][0] - pix_z[i][0], 2) +
                                pow(temp[i][1] - pix_z[i][1], 2) +
                                pow(temp[i][2] - pix_z[i][2], 2)
                                );
            }
            
            int slice_x[neighbor_size_2][3];
            int slice_y[neighbor_size_2][3];
            int slice_z[neighbor_size_2][3];
            
            if (total_x < min_x_point) {
                min_x_point = total_x;
                std::copy(&temp[0][0], &temp[0][0]+neighbor_size_2*3, &slice_x[0][0]);
            }
            
            if (total_y < min_y_point) {
                min_y_point = total_y;
                std::copy(&temp[0][0], &temp[0][0]+neighbor_size_2*3, &slice_y[0][0]);
            }
            
            if (total_z < min_z_point) {
                min_z_point = total_z;
                std::copy(&temp[0][0], &temp[0][0]+neighbor_size_2*3, &slice_z[0][0]);
            }

        }
    }
    // TODO: return the three best neighbors in the original texture, which should be located in slice_x, slice_y, and slice_z
    return ;
}

void test_neighbor() {
    srand (1234);
    int arr_x[neighbor_size_2][3] = {};
    int arr_y[neighbor_size_2][3] = {};
    int arr_z[neighbor_size_2][3] = {};
    
    for (int i = 0; i < neighbor_size_2; i += 1) {

        int ran_x = rand() % tex_size;  // random number from 0 to 255
        int ran_y = rand() % tex_size;
        arr_x[i][0] = array_R[ran_x][ran_y];
        arr_x[i][1] = array_G[ran_x][ran_y];
        arr_x[i][2] = array_B[ran_x][ran_y];
        
        ran_x = rand() % tex_size;  // random number from 0 to 255
        ran_y = rand() % tex_size;
        arr_y[i][0] = array_R[ran_x][ran_y];
        arr_y[i][1] = array_G[ran_x][ran_y];
        arr_y[i][2] = array_B[ran_x][ran_y];
        
        ran_x = rand() % tex_size;  // random number from 0 to 255
        ran_y = rand() % tex_size;
        arr_z[i][0] = array_R[ran_x][ran_y];
        arr_z[i][1] = array_G[ran_x][ran_y];
        arr_z[i][2] = array_B[ran_x][ran_y];
        
    }
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    neighborhoods (arr_x, arr_y, arr_z);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

    
}
