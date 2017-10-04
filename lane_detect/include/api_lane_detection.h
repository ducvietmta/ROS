#include "stdio.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"

#include <vector>
#include <iostream>
#include <iomanip>

#include "MSAC.h"

using namespace std;
using namespace cv;

#define MAX_NUM_LINES	30
#define VIDEO_FRAME_WIDTH 640
#define VIDEO_FRAME_HEIGHT 480


void
api_vanishing_point_init(MSAC &msac);

void edgeProcessing(Mat org, Mat &dst, Mat element, string method = "Canny");
void
api_get_vanishing_point(Mat imgGray, // input
                        Rect roi,    // input
                        MSAC &msac,  // input
                        Point &vp,    // output vnishing point
                        bool is_show_output = false,
                        string method = "Canny");
                    //Current supported method: Canny, Sobel, Prewitt, Roberts
int
laneDetect(
        cv::Mat imgGrayOrigin,
        cv::Rect rectDetect,
        vector<vector<cv::Point> > &lineSegments,
        string method = "Canny"
        );
        

void
api_get_lane_center(Mat &imgGray,
                    Point &center_point,
                    bool is_show_output = false);
