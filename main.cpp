#include "Cali_Import.h"
#include "VanPt.h"

#include "macro_define.h"

#include <ctime>

using namespace std;
using namespace cv;

Size img_size;

int main(int argc, char** argv)
{
	bool cali = true;
	if (argc >= 3)
	{
		string flag_cali = argv[2];
		if (flag_cali == "n")
			cali = false;
	}
	
	Mat cam_mtx, dist_coeff; // camera matrix(inner parameters) and distortion coefficient
	float alpha_w, alpha_h; // horizontal and vertical angle of view
	
	if (cali)
	{
		/// initialize for camera calibration
		vector<vector<Point3f> > obj_pts;
		vector<vector<Point2f> > img_pts;
		Size image_size;
		cameraCalibration(obj_pts, img_pts, image_size, 6, 9);
		
		vector<Mat> rvecs, tvecs;
		calibrateCamera(obj_pts, img_pts, image_size, cam_mtx, dist_coeff, rvecs, tvecs);
		
		cout << "cameraMatrix: " << cam_mtx << endl;
		cout << cam_mtx.at<double>(0,2) << " " << cam_mtx.at<double>(0,0) << endl;
		cout << cam_mtx.at<double>(1,2) << " " << cam_mtx.at<double>(1,1) << endl;
		
		alpha_w = atan2(cam_mtx.at<double>(0,2), cam_mtx.at<double>(0,0)); // angle of view horizontal
		alpha_h = atan2(cam_mtx.at<double>(1,2), cam_mtx.at<double>(1,1)); // angle of view vertical
	}
	else
	{
		alpha_w = 20*CV_PI/180; // default value
		alpha_h = 20*CV_PI/180; //
	}
	
	
	/// import source file
	string file_name;
	Mat image;
	VideoCapture reader;
	VideoWriter writer;
	int msc[2], ispic, video_length;
	import(argc, argv, file_name, image, reader, writer, msc, ispic, video_length);
	cout << "Set time interval: [" << msc[0] <<", "<< msc[1] << "]" << endl;
	ofstream outfile("pitch angle 80 Gabor_cluster_fit.txt");
	// ofstream outfile2("pitch angle 80 unfiltered.txt");
	
	
	/// initialize parameters that work cross frames 
	clock_t t_start = clock();
	float time_step = 0; 		// counting frames, fed to LaneImage(__nframe)
	img_size = Size(reader.get(CV_CAP_PROP_FRAME_WIDTH), reader.get(CV_CAP_PROP_FRAME_HEIGHT));

	float illu_comp = 1;
	VanPt van_pt(alpha_w, alpha_h);
	int nframe = 0;
	
	while (reader.isOpened())
	{
		reader.read(image);
		if(!image.empty() && (msc[1] == -1 || reader.get(CV_CAP_PROP_POS_MSEC) <= msc[1]))
		{
			
			cout << "current time(msc): " << reader.get(CV_CAP_PROP_POS_MSEC) << endl;
			

				Mat cali_image;
				if (cali) { undistort(image, cali_image, cam_mtx, dist_coeff); }
				else { image.copyTo(cali_image); } // not copied
				
				#ifndef NDEBUG
				// cout << "cali image size" << cali_image.size() << endl;
				// namedWindow("original color", WINDOW_AUTOSIZE);
				// imshow("original color", image);
				// namedWindow("warped color", WINDOW_AUTOSIZE);
				// imshow("warped color", cali_image);
				#endif
				
				/// get vanishing point, warp source region, illumination compensation. 
				Mat gray, warped_img;
				cvtColor(cali_image, gray, COLOR_BGR2GRAY);
				illuComp(cali_image, gray, illu_comp);
				van_pt.initialVan(cali_image, gray, warped_img);

				outputVideo(cali_image, warped_img, writer, van_pt, nframe);
				if (van_pt.ini_success)
				{
					outfile << van_pt.theta_h << endl;
					// outfile2 << van_pt.theta_h_unfil << endl;
				}
		}
		else
		{
			cout << "All frames processed. " << endl;
			outfile.close();
			// outfile2.close();
			break;
		}
	}
	return 0;
}