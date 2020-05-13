/*

 Installing opencv instructions
 https://www.codementor.io/@ohasanli/opencv-on-xcode-142qxx3sl8

 */
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/flann.hpp"
#include <iostream>
#include <vector>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace cv;
using namespace std;

// Struct for writing to .vol
struct VolumeHeader
{
    char magic[4];
    int version;
    char texName[256];
    bool wrap;
    int volSize;
    int numChannels;
    int bytesPerChannel;
};

// CONFIGURATION VARIABLE START

Ptr<Mat> img;
char input_filename[] = "/Users/tomliu/Desktop/cs184/final_proj/xcode_cplusplsu/184final/184final/input/caustic.png";
const int tex_size = 64;
// Neighborhood size
const int neighbor_size = 5;
// Tolerance for search phase ANN and accompanying parameters
const float epsilon = 2.0;
// Step for sparser voxel matching
const int stride = 2;
// Exponent for optimization phase
const float r = 0.8;

// Configuration of .vol file generation
bool tileable = false;
//char tex_name[] = "Output";
char output_filename[] = "/Users/tomliu/Desktop/cs184/final_proj/xcode_cplusplsu/184final/184final/dune.vol";

// CONFIGURATION VARIABLE END

const int neighbor_half = floor(neighbor_size / 2);
const int neighbor_size_2 = neighbor_size * neighbor_size;

// PCA for the exemplar neighborhoods
PCA PCA_analysis;
// Exemplar neighborhoods projected onto the PCA subspace
Ptr<Mat> projected_exemplar;
// Neighborhood vector to index map (used in search phase)

// Sizes for matrix initialization
int solid_sizes[] = { tex_size, tex_size, tex_size };
// 4th dimension 3 to match the x, y, z slices
int matching[tex_size * tex_size * tex_size * 3][2];

typedef cv::Vec3b Pixel;
Ptr<Mat> solid;

// Test easy solid
Ptr<Mat> solid_2;

// Initialization
void init();
// Fetching neighborhoods from the textures
Mat get_neighborhood(const Mat& input, int x, int y);
std::vector<Mat> get_slices(int x, int y, int z);
// Functions for phases of the synthesis
void optimization_phase();
void search_phase(cv::flann::GenericIndex<cvflann::L2<double>>* index, bool offset=false);
void histogram_matching();
void write_to_vol();
// Utility / Helper functions
std::vector<uchar> unroll(const Mat& m);
void weighted_avg_update_voxel(int i, int j, int k);
double l2_norm_diff(Vec3b const& u, Vec3b const& v);
Mat getPaddedROI(const Mat& input, int top_left_x, int top_left_y, int width, int height, Scalar paddingColor);
// Histogram matching
void do1ChnHist(const cv::Mat& _i, const cv::Mat* mask, double* h, double* cdf);
void histMatchRGB(cv::Mat& src, const cv::Mat* src_mask, const cv::Mat& dst, const cv::Mat* dst_mask);

void shiftRows(Mat& mat, int n);
void shiftCols(Mat& mat, int n);

int main() {
    std::cout << CV_VERSION << endl;
    std::cout << "Synthesizing solid texture from input " << input_filename << " and writing to " << output_filename << endl;
    std::clock_t start, main_start;
    double duration;
    main_start = std::clock();

    img = makePtr<Mat>(imread(input_filename));
    
    if (!(*img).data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    //imshow("image", *img);
    //waitKey(0);
    
    // START Testing Simple Textures
    
    // Stacking Texture by texture
//    solid_2 = makePtr<Mat>(3, solid_sizes, CV_8UC3);
//    for (int i = 0; i < solid_sizes[0]; i++) {
//        for (int j = 0; j < solid_sizes[1]; j++) {
//            for (int k = 0; k < solid_sizes[2]; k++){
//                Pixel ex = (*img).at<Pixel>(i, j);
//                Pixel& sol = (*solid_2).at<Pixel>(i, j, k);
//                sol[0] = ex[0];
//                sol[1] = ex[1];
//                sol[2] = ex[0];
//            }
//        }
//    }
    
//    // Shift image by 1 every layer
//    solid_2 = makePtr<Mat>(3, solid_sizes, CV_8UC3);
//    for (int i = 0; i < solid_sizes[0]; i++) {
//        for (int j = 0; j < solid_sizes[1]; j++) {
//            for (int k = 0; k < solid_sizes[2]; k++){
//                Pixel ex = (*img).at<Pixel>(j, k);
//                Pixel& sol = (*solid_2).at<Pixel>(j, k, i);
//                sol[0] = ex[0];
//                sol[1] = ex[1];
//                sol[2] = ex[0];
//            }
//        }
//        shiftRows(*img, 1);
//    }
//
//
//    std::cout << "Writing to .vol now" << endl;
//    write_to_vol();
//    return 0;
    // END
    
    // Initialize all necessary data structures including solid texture, matching
    solid = makePtr<Mat>(3, solid_sizes, CV_8UC3);
    init();
    std::cout << "First voxel value after initialization " << ((*solid).at<Pixel>(0, 0, 0)) << endl;
    // do PCA on exemplar to get downsized neighborhoods
    std::vector<std::vector<double> > neighborhood_vecs;
    for (int i = 0; i < tex_size; i++) {
        for (int j = 0; j < tex_size; j++) {
            std::vector<uchar> neighborhood = unroll(get_neighborhood((*img), i, j));
            neighborhood_vecs.push_back(std::vector<double>(neighborhood.begin(), neighborhood.end()));
        }
    }


    // Do PCA on original data
    cv::Mat PCA_data(neighborhood_vecs.size(), neighborhood_vecs.at(0).size(), CV_64F);
    for (int i = 0; i < PCA_data.rows; ++i)
        for (int j = 0; j < PCA_data.cols; ++j)
            PCA_data.at<double>(i, j) = neighborhood_vecs.at(i).at(j);
    PCA_analysis = PCA(PCA_data, Mat(), PCA::DATA_AS_ROW, 0.95);

    //projected_exemplar = makePtr<Mat>(PCA_analysis.project(PCA_data));

    // Project each vector in order
    for (int i = 0; i < neighborhood_vecs.size(); i++) {
        neighborhood_vecs[i] = PCA_analysis.project(neighborhood_vecs[i]);
    }

    // Construct mat of projected vectors
    projected_exemplar = makePtr<Mat>(neighborhood_vecs.size(), neighborhood_vecs.at(0).size(), CV_64F);
    for (int i = 0; i < (*projected_exemplar).rows; ++i)
        for (int j = 0; j < (*projected_exemplar).cols; ++j)
            (*projected_exemplar).at<double>(i, j) = neighborhood_vecs.at(i).at(j);

    

    std::cout << "First voxel value after PCA " << ((*solid).at<Pixel>(0, 0, 0)) << endl;

    
    // Initializing the ANN kdtree
    flann::KDTreeIndexParams indexParams;
    (*projected_exemplar) = (*projected_exemplar).reshape(1).t();
    Mat data_pts = (*projected_exemplar).clone();
    cout << "KDTree data size: " << data_pts.size() << endl;
    cv::flann::GenericIndex<cvflann::L2<double>> kdtree = cv::flann::GenericIndex<cvflann::L2<double>>(data_pts, cvflann::KDTreeIndexParams());

    // Run all stages for 3 iterations
    for (int i = 0; i < 3; i++) {
        cout << "Phase " << i + 1 << ":" << endl;
        optimization_phase();
        std::cout << "First voxel value after optimization " << ((*solid).at<Pixel>(0, 0, 0)) << endl;

        search_phase(&kdtree, i % 2 == 0);

        std::cout << "First voxel value after search " << ((*solid).at<Pixel>(0, 0, 0)) << endl;

        histogram_matching();

        std::cout << "First voxel value after HM " << ((*solid).at<Pixel>(0, 0, 0)) << endl;

    }
    
    
    duration = (std::clock() - (double) main_start) / (double) CLOCKS_PER_SEC;
    std::cout << "Total synthesis time: " << duration << '\n';

    std::cout << "Writing to .vol now" << endl;
    write_to_vol();
    std::cout << "Finished writing to .vol" << endl;


    /*
    // Various tests
    std::cout << "Starting exemplar neighborhood test!\n";
    //start = std::clock();
   
    
    
    //duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time to generate initial objects: " << duration << '\n';
    // Show image!
    // imshow("Image",image);
    // waitKey(0);
     std::cout << "Starting exemplar neighborhood test!\n";
    //start = std::clock();
    Mat edge, mid;
    
    edge = get_neighborhood((*img), tex_size - 1, tex_size - 1);

    //duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time to get edge neighborhood: " << duration << '\n';
    std::cout << "Edge rows: ";
    std::cout << edge.rows;
    std::cout << endl;
    std::cout << "Edge cols: ";
    std::cout << edge.cols;
    std::cout << endl;
    cout << "edge = " << endl << " " << edge << endl << endl;
    std::cout << endl;
    //start = std::clock();

    mid = get_neighborhood((*img), tex_size / 2, tex_size / 2);

    //duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time to get middle neighborhood: " << duration << '\n';

    std::cout << "Middle rows: ";
    std::cout << mid.rows;
    std::cout << endl;
    std::cout << "Middle cols: ";
    std::cout << mid.cols;
    std::cout << endl;
    cout << "middle = " << endl << " " << mid << endl << endl;
    std::cout << endl;
    std::cout << "Starting solid slices test!\n";
    //start = std::clock();

    std::vector<Mat> edge_slices = get_slices(tex_size - 1, tex_size - 1, tex_size - 1);

    //duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time to get edge slices: " << duration << '\n';
    
    for (Mat edge : edge_slices) {
        std::cout << "Edge rows: ";
        std::cout << edge.rows;
        std::cout << endl;
        std::cout << "Edge cols: ";
        std::cout << edge.cols;
        std::cout << endl;
        cout << "edge = " << endl << " " << edge << endl << endl;
        std::cout << endl;
    }
    
    //start = std::clock();
    
    std::vector<Mat> mid_slices = get_slices(tex_size / 2, tex_size / 2, tex_size / 2);
    //duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time to get middle slices: " << duration << '\n';
    for (Mat mid : mid_slices) {
        std::cout << "Middle rows: ";
        std::cout << mid.rows;
        std::cout << endl;
        std::cout << "Middle cols: ";
        std::cout << mid.cols;
        std::cout << endl;
        cout << "middle = " << endl << " " << mid << endl << endl;
        std::cout << endl;
    }
    
    //duration = (std::clock() - (double) main_start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time from start of main to end: " << duration << '\n';
    cout << "Started optimization phase!" << endl;
    start = std::clock();
    optimization_phase();
    cout << "Finished optimization phase!" << endl << endl;

    duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    std::cout << "Time for one optimization phase: " << duration << endl << endl;
    std::cout << "Testing that end neighborhood slices have changed" << endl;
    edge_slices = get_slices(tex_size - 1, tex_size - 1, tex_size - 1);
    for (Mat edge : edge_slices) {
        std::cout << "Edge rows: ";
        std::cout << edge.rows;
        std::cout << endl;
        std::cout << "Edge cols: ";
        std::cout << edge.cols;
        std::cout << endl;
        cout << "edge = " << endl << " " << edge << endl << endl;
        std::cout << endl;
    }
    std::cout << endl;
    std::cout << "Testing that middle neighborhood slices have changed" << endl;
    mid_slices = get_slices(tex_size / 2, tex_size / 2, tex_size / 2);
    for (Mat mid : mid_slices) {
        std::cout << "Middle rows: ";
        std::cout << mid.rows;
        std::cout << endl;
        std::cout << "Middle cols: ";
        std::cout << mid.cols;
        std::cout << endl;
        cout << "middle after optimization = " << endl << " " << mid << endl << endl;
        std::cout << endl;
    }
    cout << "Matching first indices before: " << matching[0][0]  << " " << matching[0][1] << endl;
    cout << "Started search phase!" << endl;
    start = std::clock();
    search_phase(&kdtree);
    cout << "Finished search phase!" << endl << endl;
    cout << "Matching first indices after: " << matching[0][0] << " " << matching[0][1] << endl;
    duration = (std::clock() - (double)start) / (double) CLOCKS_PER_SEC;
    std::cout << "Time for one search phase: " << duration << endl << endl;
    //duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time to get edge slices: " << duration << '\n';
    edge_slices = get_slices(tex_size - 1, tex_size - 1, tex_size - 1);

    for (Mat edge : edge_slices) {
        std::cout << "Edge rows: ";
        std::cout << edge.rows;
        std::cout << endl;
        std::cout << "Edge cols: ";
        std::cout << edge.cols;
        std::cout << endl;
        cout << "edge before histogram matching = " << endl << " " << edge << endl << endl;
        std::cout << endl;
    }

    //start = std::clock();

    mid_slices = get_slices(tex_size / 2, tex_size / 2, tex_size / 2);
    //duration = (std::clock() - (double) start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time to get middle slices: " << duration << '\n';
    for (Mat mid : mid_slices) {
        std::cout << "Middle rows: ";
        std::cout << mid.rows;
        std::cout << endl;
        std::cout << "Middle cols: ";
        std::cout << mid.cols;
        std::cout << endl;
        cout << "middle before histogram matching =  " << endl << " " << mid << endl << endl;
        std::cout << endl;
    }
    cout << "Solid size before: " << (*solid).size() << endl;
    cout << "Solid dims before: " << (*solid).dims << endl;
    cout << "Solid channels before: " << (*solid).channels() << endl;
    histogram_matching();
    cout << "Solid size after: " << (* solid).size() << endl;
    cout << "Solid dims after: " << (*solid).dims << endl;
    cout << "Solid channels after: " << (*solid).channels() << endl;
    std::cout << "Testing that end neighborhood slices have changed" << endl;
    edge_slices = get_slices(tex_size - 1, tex_size - 1, tex_size - 1);
    for (Mat edge : edge_slices) {
        std::cout << "Edge rows: ";
        std::cout << edge.rows;
        std::cout << endl;
        std::cout << "Edge cols: ";
        std::cout << edge.cols;
        std::cout << endl;
        cout << "edge after histogram matching =" << endl << " " << edge << endl << endl;
        std::cout << endl;
    }
    std::cout << endl;
    std::cout << "Testing that middle neighborhood slices have changed" << endl;
    mid_slices = get_slices(tex_size / 2, tex_size / 2, tex_size / 2);
    for (Mat mid : mid_slices) {
        std::cout << "Middle rows: ";
        std::cout << mid.rows;
        std::cout << endl;
        std::cout << "Middle cols: ";
        std::cout << mid.cols;
        std::cout << endl;
        cout << "middle after histogram matching = " << endl << " " << mid << endl << endl;
        std::cout << endl;
    }*/
}


void init() {

    // initialize seed
    srand(1234);
    
    // initialize solid texture with random values from exemplar and also fill random matchings
    for (int i = 0; i < solid_sizes[0]; i++) {
        for (int j = 0; j < solid_sizes[1]; j++) {
            for (int k = 0; k < solid_sizes[2]; k++){
                //int ran_x = rand() % img.cols;  // random number from 0 to 255
                //int ran_y = rand() % img.rows;
                for (int m = 0; m < 3; m++) {
                    int ran_x = rand() % (*img).cols;  // random number from 0 to 255
                    int ran_y = rand() % (*img).rows;
                    matching[3 * i + j * 3 * tex_size + k * 3 * tex_size * tex_size + m][0] = ran_x;
                    matching[3 * i + j * 3 * tex_size + k * 3 * tex_size * tex_size + m][1] = ran_y;
                    /*
                    // Random perturbation of indices to break symmetry (unnecessary if search phase is done first)
                    int offset_x = rand() % (img.cols - ran_x - 1);  // random number from 0 to 255
                    int offset_y = rand() % (img.rows - ran_y - 1;
                    
                    matching[i + j * tex_size + k * tex_size * tex_size + m][0] = img.cols, ran_x + offset_x;
                    matching[i + j * tex_size + k * tex_size * tex_size + m][1] = img.rows, ran_y + offset_y;
                    */
                }
                int ran_x = rand() % (*img).cols;  // random number from 0 to 255
                int ran_y = rand() % (*img).rows;
                Pixel ex = (*img).at<Pixel>(ran_x, ran_y);
                Pixel& sol = (*solid).at<Pixel>(i, j, k);
                if (i == tex_size / 2 && j == tex_size / 2 && k == tex_size / 2) {
                    cout << "Random exemplar pixel value " << ex << endl;
                    cout << "Middle solid pixel value " << sol << endl;
                }
                sol[0] = ex[0];
                sol[1] = ex[1];
                sol[2] = ex[0];
            }
        }
    }
}

Mat getPaddedROI(const Mat& input, int top_left_x, int top_left_y, int width, int height, Scalar paddingColor) {
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    Mat output;
    if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
        // border padding will be required
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

        if (top_left_x < 0) {
            width = width + top_left_x;
            border_left = -1 * top_left_x;
            top_left_x = 0;
        }
        if (top_left_y < 0) {
            height = height + top_left_y;
            border_top = -1 * top_left_y;
            top_left_y = 0;
        }
        if (bottom_right_x > input.cols) {
            width = width - (bottom_right_x - input.cols);
            border_right = bottom_right_x - input.cols;
        }
        if (bottom_right_y > input.rows) {
            height = height - (bottom_right_y - input.rows);
            border_bottom = bottom_right_y - input.rows;
        }

        Rect R(top_left_x, top_left_y, width, height);
        copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, BORDER_CONSTANT, paddingColor);
    }
    else {
        // no border padding required
        Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
}

// Return the neighborhood for a particular x and y given a mat
// Use img for getting from exemplar (done this way to simplify slicing)
Mat get_neighborhood(const Mat& input, int x, int y) {
    int width = neighbor_size;
    int x_start = max(0, x - neighbor_half);
    int y_start = max(0, y - neighbor_half);
    Mat nh = getPaddedROI(input, x_start, y_start, width, width, 0).clone();
    return nh;
}

// Returns slices along all three axes of the solid
std::vector<Mat> get_slices(int x, int y, int z) {
    Range xy_ranges[] = { Range::all(), Range::all(),  Range(z, z + 1) };
    Mat xy_slice3d = (*solid)(xy_ranges).clone();
    Mat xy_plane(tex_size, tex_size, CV_8UC3, xy_slice3d.data);
    Range xz_ranges[] = { Range::all(), Range(y, y + 1),  Range::all() };
    Mat xz_slice3d = (*solid)(xz_ranges).clone();
    Mat xz_plane(tex_size, tex_size, CV_8UC3, xz_slice3d.data);
    Range yz_ranges[] = { Range(x, x + 1), Range::all(),  Range::all() };
    Mat yz_slice3d = (*solid)(yz_ranges).clone();
    Mat yz_plane(tex_size, tex_size, CV_8UC3, yz_slice3d.data);
    std::vector<Mat> slices = { get_neighborhood(xy_plane, x, y), get_neighborhood(xz_plane, x, z) , get_neighborhood(yz_plane, y, z) };
    return slices;
}

std::vector<uchar> unroll(const Mat& m) {
    std::vector<uchar> array;
    if (m.isContinuous()) {
        // array.assign(mat.datastart, mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
        array.assign(m.data, m.data + m.total());
    }
    else {
        for (int i = 0; i < m.rows; ++i) {
            array.insert(array.end(), m.ptr<uchar>(i), m.ptr<uchar>(i) + m.cols);
        }
    }
    return array;
}

// Implements equation 4 from the Kopf paper. Uses matching to get matching exemplar neighborhoods by center
// and computes weights on the fly
void optimization_phase() {
    
    for (int i = 0; i < solid_sizes[0]; i++) {
        for (int j = 0; j < solid_sizes[1]; j++) {
            for (int k = 0; k < solid_sizes[2]; k++) {
                weighted_avg_update_voxel(i, j, k);
            }
        }
        if (i % 10 == 0) {
            cout << "Optimization progress: " << round(i * 100 / tex_size) << "%";
            cout << '\r';
        }
    }
    cout << "Optimization progress: DONE";
    cout << endl;
}

void weighted_avg_update_voxel(int i, int j, int k) {
    
    double w_exponent = r - 2.0;
    std::vector<Mat> slices = get_slices(i, j, k);
    double denom = .0;
    std::vector<double> total = { 0, 0, 0 };

    Pixel& voxel = (*solid).at<Pixel>(i, j, k);
    for (int s = 0; s < 3; s++) {
        int matching_x = matching[3 * i + j * 3 * tex_size + k * 3 * tex_size * tex_size + s][0];
        int matching_y = matching[3 * i + j * 3 * tex_size + k * 3 * tex_size * tex_size + s][1];
        if (i == tex_size / 2 && j == tex_size / 2 && k == tex_size / 2) {
            cout << "Middle voxel check during update:" << endl;
            cout << "Middle voxel matching indices for slice " << s << " are:" << endl << "x: " << matching_x << endl << "y: " << matching_y << endl;
        }
        Mat exemplar_nh = get_neighborhood((*img), matching_x, matching_y);
        for (int x = 0; x < neighbor_size; x++) {
            for (int y = 0; y < neighbor_size; y++) {
                Pixel exemplar_texel = exemplar_nh.at<Pixel>(x, y);
                double w = pow(l2_norm_diff(slices[s].at<Pixel>(x, y), exemplar_texel), w_exponent);
                total[0] += w * exemplar_texel[0];
                total[1] += w * exemplar_texel[1];
                total[2] += w * exemplar_texel[2];
                denom += w;
            }
        }
    }
    voxel[0] = (uchar)(total[0] / denom);
    voxel[1] = (uchar)(total[1] / denom);
    voxel[2] = (uchar)(total[2] / denom);
}

// Implements search phase from the Kopf paper. Fills matching exemplar neighborhoods by center
// by computing PCA for exemplar neighborhoods (done in init) and using ANN to get matching exemplar neighborhoods
void search_phase(cv::flann::GenericIndex<cvflann::L2<double>>* index, bool offset) {
    // Alternate boolean offset in order to touch all voxels at some point, set to True on every other search phase iteration
    // Need to alternate due to striding calculations
    int start = offset ? 1 : 0;
    for (int i = start; i < tex_size; i += stride) {
        for (int j = start; j < tex_size; j += stride) {
            for (int k = start; k < tex_size; k += stride) {
                std::vector<Mat> slices = get_slices(i, j, k);
                for (int s = 0; s < 3; s++) {
                    
                    std::vector<uchar> unrolled = unroll(slices[s]);
                    std::vector<double> voxel_nh = std::vector<double>(unrolled.begin(), unrolled.end());
                    std::vector<double> projected = PCA_analysis.project(voxel_nh);
                    /*
                    cout << "Original slice: ";
                    for (std::vector<double>::const_iterator i = voxel_nh.begin(); i != voxel_nh.end(); ++i)
                        std::cout << *i << ' ';
                    std::cout << endl;
                    
                    cout << "Projected slice: ";
                    for (std::vector<double>::const_iterator i = query.begin(); i != query.end(); ++i)
                        std::cout << *i << ' ';
                    std::cout << endl;
                    */
                    // Finding the nearest neighbor using FLANN
                    //Mat indices = Mat(1, 1, CV_32S);
                    //Mat dists = Mat(1, 1, CV_64F);
                    std::vector<int> indices(1);
                    std::vector<cv::flann::GenericIndex<cvflann::L2<double>>::DistanceType> distances(1);
                    cvflann::SearchParams params(32, epsilon, true);
                    Mat query = Mat(projected).reshape(1);
                    
                    //cout << "Query data size: " << query.size() << endl;
                    index->knnSearch(query, indices, distances, 1, params);
                    // Recover original
                    // Since index is indexing into all neighborhood vectors from exemplar
                    
                    matching[3 * i + j * 3 * tex_size + k * 3 * tex_size * tex_size + s][0] = indices[0] % tex_size;//indices.at<int>(0, 0) % tex_size;
                    matching[3 * i + j * 3 * tex_size + k * 3 * tex_size * tex_size + s][1] = floor(indices[0] / tex_size);//floor(indices.at<int>(0, 0) / tex_size);

                    if (i == tex_size / 2 && j == tex_size / 2 && k == tex_size / 2) {
                        cout << "Middle voxel check during search:" << endl;
                        cout << "Middle voxel slice " << s << " is ";
                        for (auto i = unrolled.begin(); i != unrolled.end(); ++i)
                            std::cout << (double) *i << ' ';
                        cout << endl;
                        cout << "Middle voxel projected " << s << " is ";
                        for (auto i = projected.begin(); i != projected.end(); ++i)
                            std::cout << *i << ' ';
                        cout << endl;
                        cout << "Query size: " << query.size() << endl;
                        cout << "index for the matching vector is " << indices[0] <<endl;
                        //cout << "projected exemplar is = " << endl << " " << (*projected_exemplar) << endl << endl;
                        cout << "Matching PCA exemplar vector is ";
                        for (auto i = indices[0]; i != indices[0] + 3; ++i)
                            std::cout << *((double*)(*projected_exemplar).data + i) << ' ';
                        cout << endl;
                    }
                }
            }
        }
        if (i % 10 == 0 && !offset || i % 10 == 1 && offset) {
            cout << "Search progress: " << round(i * 100 / tex_size) << "%";
            cout << '\r';
        }
    }
    cout << "Search progress: DONE";
    cout << endl;
}

void histogram_matching() {
    histMatchRGB((*solid), NULL, (*img), NULL);
    cout << "Histogram matching progress: DONE";
    cout << endl;
}


// Match Red, Green and Blue histograms of 'src' to that of 'dst', according to both masks.
// Based on: http://www.morethantechnica...
// Modified by Shervin Emami so it can also pass NULL for the masks if you want to process the whole image.
void histMatchRGB(cv::Mat& src, const cv::Mat* src_mask, const cv::Mat& dst, const cv::Mat* dst_mask)
{
    std::vector<Mat> chns;
    cv::split(src, chns);
    std::vector<Mat> chns1;
    cv::split(dst, chns1);
    cv::Mat src_hist = cv::Mat::zeros(1, 256, CV_64FC1);
    cv::Mat dst_hist = cv::Mat::zeros(1, 256, CV_64FC1);
    cv::Mat src_cdf = cv::Mat::zeros(1, 256, CV_64FC1);
    cv::Mat dst_cdf = cv::Mat::zeros(1, 256, CV_64FC1);
    cv::Mat Mv(1, 256, CV_8UC1);
    uchar* M = Mv.ptr();

    for (int i = 0; i < 3; i++) {
        src_hist.setTo(Scalar(0));
        dst_hist.setTo(Scalar(0));
        src_cdf.setTo(Scalar(0));
        src_cdf.setTo(Scalar(0));

        double* _src_cdf = (double*) src_cdf.ptr();
        double* _dst_cdf = (double*) dst_cdf.ptr();
        double* _src_hist = (double*) src_hist.ptr();
        double* _dst_hist = (double*) dst_hist.ptr();

        do1ChnHist(chns[i], src_mask, _src_hist, _src_cdf);
        do1ChnHist(chns1[i], dst_mask, _dst_hist, _dst_cdf);

        uchar last = 0;
        double const HISTMATCH_EPSILON = 0.000001;

        for (int j = 0; j < src_cdf.cols; j++) {
            double F1j = _src_cdf[j];

            for (uchar k = last; k < dst_cdf.cols; k++) {
                double F2k = _dst_cdf[k];
                // Note: Two tests were combined into one for efficiency, by Shervin Emami, Apr 24th 2011.
                //if (abs(F2k - F1j) F1j) {
                if (F2k > F1j - HISTMATCH_EPSILON) {
                    M[j] = k;
                    last = k;
                    break;
                }
            }
        }

        cv::Mat lut(1, 256, CV_8UC1, M);
        cv::LUT(chns[i], lut, chns[i]);
    }

    cv::Mat res;
    cv::merge(chns, res);

    res.copyTo(src);
}

// Compute histogram and CDF for an image with mask
void do1ChnHist(const cv::Mat& _i, const cv::Mat* mask, double* h, double* cdf)
{
    cv::Mat _t = _i.reshape(1, 1);

    // Get the histogram with or without the mask
    if (mask) {
        cv::Mat _tm;
        mask->copyTo(_tm);
        _tm = _tm.reshape(1, 1);
        for (int p = 0; p < _t.cols; p++) {
            uchar m = _tm.at<uchar>(0, p);
            if (m > 0) { // Mask value
                uchar c = _t.at<uchar>(0, p); // Image value
                h[c] += 1.0; // Histogram
            }
        }
    }
    else {
        for (int p = 0; p < _t.cols; p++) {
            uchar c = _t.at<uchar>(0, p); // Image value
            h[c] += 1.0; // Histogram
        }
    }

    //normalize hist to a max value of 1.0
    cv::Mat _tmp(1, 256, CV_64FC1, h);
    double minVal, maxVal;
    cv::minMaxLoc(_tmp, &minVal, &maxVal);
    _tmp = _tmp / maxVal;

    // Calculate the Cumulative Distribution Function
    cdf[0] = h[0];
    for (int j = 1; j < 256; j++) {
        cdf[j] = cdf[j - 1] + h[j];
    }

    //normalize CDF to a max value of 1.0
    _tmp.data = (uchar*)cdf; // Array of doubles, but gets a byte pointer.
    cv::minMaxLoc(_tmp, &minVal, &maxVal);
    _tmp = _tmp / maxVal;
}

void write_to_vol() {

    struct VolumeHeader* header = (struct VolumeHeader *) calloc(4096, 1);
    strncpy(header->magic, "VOLU", 4);
    header->version = 4;
    strcpy(header->texName, input_filename);
    header->wrap = tileable;
    header->volSize = tex_size;
    header->numChannels = 3;
    header->bytesPerChannel = 1;

    FILE* pFile;
    pFile = fopen(output_filename, "wb");

    cout << "BGR" << ((*solid_2).at<Pixel>(0, 0, 0)) << endl;

    // Convert from BGR to RGB
     /*int from_to[] = { 0,2, 1,1, 2,0 };
    Mat solid_rgb((*solid).size(), (*solid).type());
    cv::mixChannels(solid, 1, &solid_rgb, 1, from_to, 3);
    unsigned char* buffer = solid_rgb.data;*/
    cv::Mat rgb(3, solid_sizes, CV_8UC3);
    int from_to[] = { 0,2, 1,1, 2,0 };
    cv::mixChannels(solid_2, 1, &rgb, 1, from_to, 3);
    unsigned char* buffer = rgb.data;
    //cv::cvtColor((*solid), (*solid), BGR2RGB);
    //unsigned char* buffer = (*solid).data;
    
    cout << "RGB" << (rgb.at<Pixel>(0, 0, 0)) << endl;

    // Writing header to file
    //fwrite(header, sizeof(char), sizeof(struct VolumeHeader), pFile);
    fwrite(header, sizeof(char), 4096, pFile);
    /*
    // Zero padding
    const int pad_size = 4096 - sizeof(struct VolumeHeader);
    char zeros[pad_size];
    fwrite(zeros, sizeof(char), pad_size, pFile);
    */
    // Write the data itself
    fwrite(buffer, sizeof(char), pow(tex_size, 3) * header->numChannels * header->bytesPerChannel, pFile);
    fclose(pFile);
}


double l2_norm_diff(Vec3b const& u, Vec3b const& v) {
    int accum = 0.;
    for (int i = 0; i < 3; ++i) {
        int diff = u[i] - v[i];
        accum += diff * diff;
    }
    return sqrt(accum);
}

// src https://stackoverflow.com/questions/10420454/shift-like-matlab-function-rows-or-columns-of-a-matrix-in-opencv
void shiftRows(Mat& mat) {
    Mat temp;
    Mat m;
    int k = (mat.rows-1);
    mat.row(k).copyTo(temp);
    for(; k > 0 ; k--) {
        m = mat.row(k);
        mat.row(k-1).copyTo(m);
    }
    m = mat.row(0);
    temp.copyTo(m);
}

void shiftRows(Mat& mat, int n) {
    for(int k=0; k < n;k++) {
        shiftRows(mat);
    }
}

void shiftCols(Mat& mat, int n) {
    transpose(mat,mat);
    shiftRows(mat,n);
    transpose(mat,mat);
}

