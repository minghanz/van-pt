#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor, Canny
#include <opencv2/calib3d.hpp> // for calibration
#include <opencv2/ml/ml.hpp> // for machine learning
// #include <opencv2/tracking.hpp> // Kalman
//#include <armadillo>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp

#define NDEBUG_CL // COLOR
#define NDEBUG_GR // GRADIENT
#define NDEBUG_IN // INITIAL
#define NDEBUG_TR // TRAIN
#define NDEBUG
//#define NTIME
#define HIGH_BOT

#define DTREE
//#define LREGG

//#define EVA

//#define COUT

#define DRAW

#define CALI_VAN // whether prior calibrated vanishing point is available

// #define CLUSTER_FOR_VOTE //  NMS+Hough  or  cluster+fit  in GaborVote
#define CANNY_VOTE