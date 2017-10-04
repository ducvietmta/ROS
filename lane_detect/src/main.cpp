/*--- ROS includes ---*/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "std_msgs/Int32.h"
/*--- OpenCV includes ---*/
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "api_lane_detection.h"
#include "SignDetection.h"
#include "SignRecognition.h"

#define VIDEO_FRAME_WIDTH 640
#define VIDEO_FRAME_HEIGHT 480
int frame_width = VIDEO_FRAME_WIDTH;
int frame_height = VIDEO_FRAME_HEIGHT;
Point carPosition(frame_width / 2, frame_height);
Point prvPosition = carPosition;
Point center_point(0,0);
ros::Publisher point_publisher;
using namespace std;

// 1. Topics this node subscribes to
static const string CAMERA1_TOPIC = "/camera/rgb/image_raw";
static const string CAMERA2_TOPIC = "/camera/depth/image_raw";

// Names for the two camera windows
static const string OPENCV_WINDOW1 = "Color Image Window";
static const string OPENCV_WINDOW2 = "Depth Image Window";

cv::Mat src1, src2;
cv::Mat depthImg, colorImg;
MSAC msac;
std_msgs::Int32 message;
class ImageConverter{
    private:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        image_transport::Subscriber image_sub_, image_sub2_;

    public:
        // constructor
        ImageConverter() :
        it_(nh_){
            image_sub_ = it_.subscribe(CAMERA1_TOPIC, 1, &ImageConverter::imageCb, this);
            image_sub2_ = it_.subscribe(CAMERA2_TOPIC, 1, &ImageConverter::imageCb2, this);
        }
        ~ImageConverter(){
            cv::destroyAllWindows();
        }
        double getTheta(Point car, Point dst) {
            if (dst.x == car.x) return 0;
            if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
            double pi = acos(-1.0);
            double dx = dst.x - car.x;
            double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
            if (dx < 0) return -atan(-dx / dy) * 180 / pi;
            return atan(dx / dy) * 180 / pi;
        }
  /*-------- Callback functions for color camera  -----------*/
        void imageCb(const sensor_msgs::ImageConstPtr& msg){
            cv::Mat grayImage;
            cv_bridge::CvImagePtr cv_ptr;
            try{
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception& e){
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
	        cv::cvtColor(cv_ptr->image, colorImg, cv::COLOR_BGR2RGB);
	        cv::cvtColor(cv_ptr->image, grayImage, cv::COLOR_BGR2GRAY);
	        cv::Rect roi1 = cv::Rect(0, VIDEO_FRAME_HEIGHT*3/4,VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT/4);
            api_get_vanishing_point( grayImage, roi1, msac, center_point, false,"Wavelet");
	        if (center_point.x == 0 && center_point.y == 0) center_point = prvPosition;
	        prvPosition = center_point;
	        double angDiff = getTheta(carPosition, center_point);
            double theta = -angDiff;
	        cv::circle(cv_ptr->image, center_point, 4, cv::Scalar(0, 255, 255), 3);
	        cv::imshow(OPENCV_WINDOW1, cv_ptr->image);
	        cv::waitKey(1);
	        message.data = theta;
	        ROS_INFO("%d",message.data);
	        point_publisher.publish(message);
        }

  /*-------- Callback functions for depth camera  -----------*/
        void imageCb2(const sensor_msgs::ImageConstPtr& msg){
            cv_bridge::CvImagePtr cv_ptr;
	        try{
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            }
            catch (cv_bridge::Exception& e){
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
	        normalize(cv_ptr->image, depthImg, 255, 0, cv::NORM_MINMAX);
	        depthImg.convertTo(depthImg, CV_8U);
            cv::imshow(OPENCV_WINDOW2, depthImg);
            cv::waitKey(3);
        }
};
/*---- main function ----*/
int main(int argc, char** argv)
{
	api_vanishing_point_init( msac );
	ros::init(argc, argv, "image_converter");
	ros::NodeHandle node_obj;
	point_publisher = node_obj.advertise<std_msgs::Int32>("center_point",1000);	
	ImageConverter ic;
	ros::spin();
	return 0; 
}
