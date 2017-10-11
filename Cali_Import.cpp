#include "Cali_Import.h"
#include <cstdio>

using namespace std;
using namespace cv;

float ym_per_pix = 40./500.;
float xm_per_pix = 3.7/200.;

int warp_col = 400;
int warp_row = 500;

void cameraCalibration(vector<vector<Point3f> >& obj_pts, vector<vector<Point2f> >& img_pts, Size& image_size, int ny, int nx)
{
	vector<Point3f> objp;
	vector<Point2f> imgp;
	
	// string first_file = "../usb_cali_images/cali%d.jpg"; //"../camera_cal/calibration%d.jpg";
	string first_file = "../camera_cal/calibration%d.jpg";
	VideoCapture calib_imgseq(first_file);
	if (!calib_imgseq.isOpened())
	{
	    cout  << "Could not open the calibration image sequence: " << first_file << endl;
	    return;
	}
	
	for(int i = 0; i<ny*nx; i++)
	{
		objp.push_back(Point3f(i%nx, i/nx, 0)); 
		/*
		#ifndef NDEBUG
		cout << "objp: " << objp[i] << endl;
		#endif
		*/
	}
	
	Mat calib_img, gray;
	Size patternsize(nx, ny);
	for (;;)
	{
		calib_imgseq.read(calib_img);
		if (calib_img.empty()) break;
		image_size = calib_img.size();
		cvtColor(calib_img, gray, COLOR_BGR2GRAY);
		bool patternfound = findChessboardCorners(gray, patternsize, imgp);
		
		if (patternfound)
		{
			obj_pts.push_back(objp);
			img_pts.push_back(imgp);
			/*
			#ifndef NDEBUG
			drawChessboardCorners(calib_img, patternsize, imgp, patternfound);
			namedWindow("calib-image", WINDOW_AUTOSIZE);
			imshow("calib-image", calib_img);
			waitKey(0);
			#endif
			*/
		}
	}
	return;
}


void import(int argc, char** argv, string& file_name, Mat& image, VideoCapture& reader, VideoWriter& writer, int msc[], int& ispic, int& video_length)
{
	if (argc>1) file_name = argv[1];
	else file_name = "../challenge.avi";
	
	if (file_name == "camera")
	{
		reader.open(0);              // Open input
	    if (!reader.isOpened())
	    {
	        cout  << "Could not open the input video: " << file_name << endl;
	        return;
	    }
	    cout << "Camera stream opened successfully. " << endl;
	    
	    int codec = static_cast<int>(reader.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
		Size S = Size((int) reader.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
               (int) reader.get(CAP_PROP_FRAME_HEIGHT));
                  
		const string out_name = file_name + "_res.avi";
		writer.open(out_name, codec, reader.get(CAP_PROP_FPS), S, true);
		cout << codec << " " << S << " " << reader.get(CAP_PROP_FPS) << endl;
	    return;
	}
	
	// check the format of source file
	const string::size_type dotpos = file_name.find_last_of('.');
	const string::size_type slashpos = file_name.find_last_of('/');
	const string extension = file_name.substr(dotpos+1);
	size_t percpos = file_name.find('%');
	if((extension == "jpg" || extension == "png" )&& percpos == string::npos)
	{
		ispic = 1;
		cout << "Format: picture." << endl;
		image = imread(file_name, IMREAD_COLOR);
		if(image.empty())
		{
			cout << "Could not open or find the image." << endl;
			return;
		}
	}
	else if(extension == "avi" || ( (extension == "jpg" ||extension == "png") && percpos != string::npos ) )
	{
		ispic = 0;
		cout << "Format: video." << endl;
		
		reader.open(file_name);              // Open input
	    if (!reader.isOpened())
	    {
	        cout  << "Could not open the input video: " << file_name << endl;
	        return;
	    }
	    
        string out_name;
        if (extension == "avi")
        {
			video_length = (int) reader.get(CAP_PROP_FRAME_COUNT);
			// retrieve the property of source video
			int codec = static_cast<int>(reader.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
			Size S = Size((int) reader.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) reader.get(CAP_PROP_FRAME_HEIGHT));
                  
			const string out_name = file_name.substr(slashpos+1,dotpos-slashpos-1) + "2_res.avi";
			writer.open(out_name, codec, reader.get(CAP_PROP_FPS), S, true);
			cout << codec << " " << S << " " << reader.get(CAP_PROP_FPS) << endl;
		}
		else
		{
			int codec = 1196444237;
			double fps = 10;
			Size S = Size((int) reader.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) reader.get(CAP_PROP_FRAME_HEIGHT));
            
            const string out_name = file_name.substr(slashpos+1,percpos-slashpos-1) + "_res.avi";
            writer.open(out_name, codec, fps, S, true);
			cout << codec << " " << S << " " << fps << endl;
		}
		// construct the output class
		
        if (!writer.isOpened())
		{
			cout  << "Could not open the output video for write: " << out_name << endl;
			return;
		}
		
		if (argc == 4)
		{
			msc[0] = (int)strtol(argv[3], NULL, 10);
			msc[1] = (int)strtol(argv[3], NULL, 10);
			reader.set(CAP_PROP_POS_MSEC, msc[0]);  
		}
		else if (argc == 5)
		{
			msc[0] = (int)strtol(argv[3], NULL, 10);
			msc[1] = (int)strtol(argv[4], NULL, 10);
			reader.set(CAP_PROP_POS_MSEC, msc[0]);  
		}
		else
		{
			msc[0] = -1;
			msc[1] = -1;
		}
	}
	else
	{
		cout << "file format is not supported." << endl;
		return;
	}	
}

string x2str(int num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}
string x2str(float num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}
string x2str(double num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}
string x2str(bool num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}

void illuComp(Mat& raw_img, Mat& gray, float& illu_comp)
{
	int mode = 2; // 1: top 5%; 2: 20%-80% sample 10%
	
	/// histogram for reference white
	int hist_size = 256;
	float range_bgrls[] = {0, 256};
	const float* range_5 = {range_bgrls};
	int hist_size_h = 180;
	float range_h[] = {0, 180};
	const float* range_1 = {range_h};
	bool uniform = true;
	bool accumulate = false;
	
	Mat pixhis_gray;
	Mat height_mask_2(gray.size(), CV_8UC1, Scalar(255));
	#ifndef HIGH_BOT
	height_mask_2.rowRange(0, gray.rows*2/3) = 0;
	#else
	height_mask_2.rowRange(0, gray.rows/2) = 0;
	height_mask_2.rowRange(gray.rows*7/10, gray.rows) = 0;  // for caltech data
	#endif
	
	calcHist( &gray, 1, 0, height_mask_2, pixhis_gray, 1, &hist_size, &range_5, uniform, accumulate );
	float accu_num_pix = 0;
	float accu_graylevel = 0;
	#ifndef HIGH_BOT
	float total_pix = gray.rows * gray.cols/3 ; 
	#else
	float total_pix = gray.rows * gray.cols/10*2 ; // for caltech data
	#endif
	
	if (mode == 1)
	{
		for (int i = 0; i < hist_size; i++)
		{
			accu_num_pix += pixhis_gray.at<float>(hist_size - 1 - i);
			accu_graylevel += pixhis_gray.at<float>(hist_size - 1 - i) * (hist_size - 1 - i);
			if (accu_num_pix >= 0.05*total_pix)
				break;
		}
		float base_white = accu_graylevel/accu_num_pix;
		cout << "base_white value: " << base_white << endl;
		illu_comp = 100/base_white;
	}
	else
	{
		bool start = false, ending = false;
		int limit_low = 0, limit_high = 0;
		float effective_num_pix = 0;
		for (int i = 0; i < hist_size; i++)
		{
			if ( start == false && accu_num_pix >= 0.2*total_pix )
			{
				limit_low = i;
				start = true;
				effective_num_pix = accu_num_pix;
				cout << "effective_num_pix 1: " << effective_num_pix << endl;
			}
			if ( accu_num_pix >= 0.8*total_pix && ending == false )
			{
				limit_high = i;
				ending = true;
				cout << "effective_num_pix 2: " << effective_num_pix << endl;
				effective_num_pix = accu_num_pix - effective_num_pix;
				cout << "accu_num_pix: " << accu_num_pix << endl;
				cout << "effective_num_pix 3: " << effective_num_pix << endl;
				break;
			}
			accu_num_pix += pixhis_gray.at<float>(i);
			if (start == true && ending == false)
				accu_graylevel += pixhis_gray.at<float>(i) * i;
		}
		float avg_gray = accu_graylevel / effective_num_pix;
		//cout << pixhis_gray << endl;
		cout << "total_pix: " << total_pix << ", accu_num_pix: " << accu_num_pix << endl;
		cout << accu_graylevel << " " << effective_num_pix << endl;
		cout << "avg_gray: " << avg_gray<< endl;
		illu_comp = 100 / avg_gray; // 100 80
	}
	
	#ifndef NDEBUG_IN
	// imshow("image before compen", raw_img);
	#endif
	gray = gray*illu_comp;
	raw_img = raw_img*illu_comp;
	#ifndef NDEBUG_IN
	// imshow("image after compen", raw_img);
	// waitKey(0);
	#endif
	return;
}
