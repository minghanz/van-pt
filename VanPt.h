#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor, Canny
#include <opencv2/calib3d.hpp> // for calibration
#include <opencv2/ml/ml.hpp> // for machine learning
//#include <armadillo>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <cmath>

#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp

class VanPt
{
public:
    VanPt(float alpha_w, float alpha_h);
    void initialVan(Mat color_img, Mat gray, Mat& warped_img);
    void SteerFilter(Mat image, Mat& steer_resp, Mat& steer_angle_max, Mat& steer_resp_weight);
    void getSteerKernel(Mat& kernel_x, Mat& kernel_y, Mat& kernel_xy, int ksize, double sigma);
    void GaborFilter(Mat image, Mat& gabor_resp_mag, Mat& gabor_resp_dir, Mat& gabor_weight);
    bool GaborVote(Mat gabor_resp_dir, Mat gabor_weight, Mat& gabor_vote, Mat edges);
    void NMS(Mat matrix, Mat& matrix_nms);
    void LaneDirec(Mat steer_resp_mag, Mat edges, Mat& blur_edges);
    void RoughLaneDirec(Mat steer_resp_mag, Mat& mask_side, int direction);
    
    
public:
    Point2f van_pt_ini;
    Point2f van_pt_img;
    Point2f van_pt_best;
    Point2f van_pt_avg;
    Point van_pt_best_int;
    Point2f van_pt_cali;

    vector<Vec4i> lines_vote;

    vector<Point2f> warp_src;
    vector<Point2f> warp_dst;
    vector<vector<Point> > warp_test_vec; // used in drawContour
    int y_bottom_warp;
    int y_bottom_warp_max;

	Mat per_mtx;
    Mat inv_per_mtx;

    Mat vote_lines_img;

    bool ini_flag;
    bool first_sucs;
    bool sucs_before;
    int fail_ini_count;

    bool ini_success;

    #ifdef CANNY_VOTE
    Point2f van_pt_obsv;
    float max_weight_left;
    float max_weight_right;
    float confidence;
    float conf_weight, conf_mse, conf_dist;
    // KalmanFilter kalman;
    float conf_c_x, conf_gamma_x, conf_c_y, conf_gamma_y;
    float conf_c_x_max, conf_c_x_min;
    float conf_c_y_min;
    float conf_gamma_e, conf_c_e;

    bool edgeVote(Mat image, Mat edges);
    int checkValidLine(Vec4i line);
    float getLineWeight(Vec4i line);
    float getConfidence(const vector<Point2f>& van_pt_candi, const vector<float>& van_pt_candi_w, 
        const vector<float>& valid_lines_w_left, const vector<float>& valid_lines_w_right); //, Point2f van_pt_obsv);
    void updateFlags();
    void updateTrackVar();
    #endif

	float theta_w;	// yaw angle
    float theta_h;	// pitch angle
    float theta_w_unfil, theta_h_unfil;
    const float ALPHA_W;
    const float ALPHA_H;
};

void outputVideo(Mat image, Mat warped_img, VideoWriter& writer, const VanPt& van_pt, int& nframe);
template <class T> string x2str(T num);
