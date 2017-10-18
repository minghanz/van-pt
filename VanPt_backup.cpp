bool VanPt::edgeVote(Mat image, Mat edges) // the original version as in 4
{
	/// vote for vanishing point based on Hough
	vector<Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI/180, 10, 10, 10 );
	//HoughLinesP(edges, lines, 20, CV_PI/180*3, 30, 10, 50 );

	if (lines.size() <= 0) /// safe return
	{
		cout << "Initilization failed: no Hough lines found. " << endl;
		return false;
	}

	Mat vote_left(image.rows, image.cols, CV_64FC1, Scalar(0));
	Mat vote_right(image.rows, image.cols, CV_64FC1, Scalar(0));
	Mat vote_line(image.rows, image.cols, CV_8UC1, Scalar(0));  // contain lines qualified to vote, decide bottom and sample color threshold 

	float y_bottom_left = 0, y_bottom_right= 0; // warp source bottom

	#ifndef HIGH_BOT
	float y_bottom_max = min(image.rows * 14/15, image.rows -1 );
	#else
	float y_bottom_max = min(image.rows *7/10, image.rows -1 );  // for caltech data
	#endif

		
	for (int i = 0; i < lines.size(); i++)
	{
		float x1 = lines[i][0];
		float y1 = lines[i][1];
		float x2 = lines[i][2];
		float y2 = lines[i][3];
		double w = (abs(x1-x2) +abs(y1-y2));
		float k = (x1-x2)/(y1-y2);
		
		if (abs(k) < 0.3 || abs(k) > 3 )
		{
			line(vote_lines_img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,255,0), 1);
			continue;
		}
		float x0, y0; // find lower point
		if (y1 > y2)
		{
			x0 = x1;
			y0 = y1;
		}
		else
		{
			x0 = x2;
			y0 = y2;
		}
		if (k > 0) 	// right
		{
			if (x0 < image.cols / 2)
			{
				line(vote_lines_img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,255,0), 1);
				continue;
			}
			for (int j = 0; j < y0 ; j++ )
			{
				int x_cur = x0 - k*j;
				int y_cur = y0 - j;
				if (x_cur > image.cols - 1 || x_cur < 0)
					break;
				vote_right.at<double>(y_cur, x_cur)+= w;
			}
			if (x0 + k*(y_bottom_max - y0)> image.cols - 1) // not approaching bottom
			{
				float lower_y = y0 + (image.cols - 1 - x0)/k;
				if (lower_y > y_bottom_right)
					y_bottom_right = lower_y;
			}
			else
				y_bottom_right = y_bottom_max;
			line(vote_line, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255), 1);
			line(vote_lines_img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 1);
			lines_vote.push_back(lines[i]);
		}
		else // left
		{
			if (x0 > image.cols / 2)
			{
				line(vote_lines_img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,255,0), 1);
				continue;
			}
			for (int j = 0; j < y0 ; j++ )
			{
				int x_cur = x0 - k*j;
				int y_cur = y0 - j;
				if (x_cur > image.cols - 1 || x_cur < 0)
					break;
				vote_left.at<double>(y_cur, x_cur)+= w;
			}
			if (x0 + k*(y_bottom_max - y0)< 0) // not approaching bottom
			{
				float lower_y = y0 + (0 - x0)/k;
				if (lower_y > y_bottom_left)
					y_bottom_left = lower_y;
			}
			else
				y_bottom_left = y_bottom_max;
			line(vote_line, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255), 1);
			line(vote_lines_img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 1);
			lines_vote.push_back(lines[i]);
		}
	}
	/// calculate proper warp region (lower vertex)
		y_bottom_warp = y_bottom_max ;
		if ((y_bottom_left != y_bottom_max || y_bottom_right != y_bottom_max ) && y_bottom_left != 0 && y_bottom_right != 0)
		{
			y_bottom_warp = min(y_bottom_left, y_bottom_right);
		}
	cout << "y_bottom_warp: " << y_bottom_warp << endl;


	Mat votemap = vote_left.mul(vote_right)/(vote_left + vote_right);

	double maxval;
	Point van_pt_int;
	minMaxLoc(votemap, NULL, &maxval, NULL, &van_pt_int);
	if (maxval > 0)
	{
		van_pt_ini.x = van_pt_int.x;
		van_pt_ini.y = van_pt_int.y;
		ini_success = true;
	}
	else
	{
		ini_success = false;
		return false;
	}

	theta_w = atan(tan(ALPHA_W)*((van_pt_ini.x - img_size.width/2)/(img_size.width/2))); 	// yaw angle 
	theta_h = atan(tan(ALPHA_H)*((van_pt_ini.y - img_size.height/2)/(img_size.height/2)));	// pitch angle

	cout << "first van pt: " << van_pt_ini << "maxval: "<< maxval << endl;

	#ifndef NDEBUG_IN
	int thickness = ini_success ? -1:2;
	Mat show_garbor;
	normalize(vote_left+vote_right, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
	circle(show_garbor, Point(van_pt_ini), 5, Scalar( 255), thickness);
	imshow("vote_left_right", show_garbor);
	// imshow("vote_left", vote_left);
	// imshow("vote_right", vote_right);
	normalize(edges, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
	circle(show_garbor, Point(van_pt_ini), 5, Scalar( 255), thickness);
	imshow("gabor_weight", show_garbor);
	circle(vote_line, Point(van_pt_ini), 5, Scalar( 255), thickness);
	imshow("vote_line", vote_line);
	// waitKey(0);
	#endif

	return true;
}