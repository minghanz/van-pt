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

#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp



void cameraCalibration(vector<vector<Point3f> >& obj_pts, vector<vector<Point2f> >& img_pts, Size& image_size, int ny=6, int nx=9);
void import(int argc, char** argv, string& file_name, Mat& image, VideoCapture& reader, VideoWriter& writer, int msc[], int& ispic, int& video_length);

void illuComp(Mat& raw_img, Mat& gray, float& illu_comp);


string x2str(int num);
string x2str(float num);
string x2str(double num);
string x2str(bool num);
