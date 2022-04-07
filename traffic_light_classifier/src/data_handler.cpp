#include "ros/ros.h"
#include <iostream>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include "opencv2/opencv.hpp"
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char **argv){

    ros::init(argc, argv, "data_handler");

    ros::NodeHandle n;

    ros::Publisher data_pub = n.advertise<sensor_msgs::Image>("data_handler", 1000);
    ros::Rate loop_rate(100);

    std::string const root_path = ros::package::getPath("traffic_light_classifier");
    cv::VideoCapture cap(root_path + "/video.mp4"); 
    if(!cap.isOpened()){
	    std::cout << "Error opening video stream or file" << std::endl;
	    return -1;
	  }

    uint32_t counter = 0;
    
    while (ros::ok())
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
	        break;
        cv::cvtColor( frame, frame, cv::COLOR_BGR2RGB );
        
        std_msgs::Header header; // empty header
        header.seq = counter; // user defined counter
        header.stamp = ros::Time::now();

        cv_bridge::CvImage out_msg;
        out_msg.header = header;
        out_msg.encoding = sensor_msgs::image_encodings::RGB8;
        out_msg.image = frame;     
        
        counter++;

        data_pub.publish(out_msg.toImageMsg());

        ros::spinOnce();

        loop_rate.sleep();
    }

    cap.release();

    return 0;
}