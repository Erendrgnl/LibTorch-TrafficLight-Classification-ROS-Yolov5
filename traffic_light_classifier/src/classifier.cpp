#include "ros/ros.h"
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include <torch/script.h>
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <iostream>
#include <cv_bridge/cv_bridge.h>

//Function Declaration
std::vector<torch::Tensor> non_max_suppression(torch::Tensor, float, float);
std::vector<torch::Tensor> object_detection(torch::jit::script::Module, cv::Mat);
void draw_traffic_light(torch::jit::script::Module, cv::Mat& ,std::vector<torch::Tensor>, std::vector<std::string> );
int classify_traffic_light(torch::jit::script::Module , cv::Mat );

class YOLOv5ROS
{
    public:
        YOLOv5ROS()
        {
            pub_ = n_.advertise<sensor_msgs::Image>("/detection_result", 1000);
            sub_ = n_.subscribe("/data_handler", 1, &YOLOv5ROS::callback, this);
            
            // Load Models
            yolo_model = torch::jit::load(root_path+"/yolov5s.torchscript.pt");
            resnet18 = torch::jit::load(root_path+"/resnet18_jit.pt");

            // Load Coco Names
            std::ifstream f(root_path+"/coco.names");
            std::string name = "";
            while (std::getline(f, name))
            {
                classnames.push_back(name);
            }
        }

    void callback(const sensor_msgs::Image::ConstPtr& image_data)
    {
        cv_bridge::CvImagePtr cv_ptr;
        std::vector<torch::Tensor> detections;
        
        cv_ptr = cv_bridge::toCvCopy(image_data, sensor_msgs::image_encodings::RGB8);
        cv::Mat image = cv_ptr->image;
        
        detections = object_detection(yolo_model,image);
        draw_traffic_light(resnet18,image,detections,classnames);        
        
        cv_bridge::CvImage detection_result;
        std_msgs::Header header; // empty header
        header = cv_ptr->header;
        header.stamp = ros::Time::now();
        detection_result.header = header;
        detection_result.encoding = sensor_msgs::image_encodings::RGB8;
        detection_result.image = image;

        pub_.publish(detection_result);
    }

    private:
        ros::NodeHandle n_;
        ros::Publisher pub_;
        ros::Subscriber sub_;
        torch::jit::script::Module yolo_model;
        torch::jit::script::Module resnet18;
        std::vector<std::string> classnames;
        std::string const root_path = ros::package::getPath("traffic_light_classifier");;
};

int main(int argc, char **argv){

    ros::init(argc, argv, "yolov5");

    YOLOv5ROS yolov5_ros;

    ros::spin();

    return 0;
}

std::vector<torch::Tensor> object_detection(torch::jit::script::Module module, cv::Mat frame)
{
    // Preparing input tensor
    cv::Mat img;
    cv::resize(frame, img, cv::Size(640, 384));
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);
    imgTensor = imgTensor.permute({2,0,1});
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    imgTensor = imgTensor.unsqueeze(0);    

    torch::Tensor preds = module.forward({imgTensor}).toTuple()->elements()[0].toTensor();
    std::vector<torch::Tensor> dets = non_max_suppression(preds, 0.4, 0.5);
    return dets;
}

int classify_traffic_light(torch::jit::script::Module module, cv::Mat frame)
{
    cv::Mat img;
    cv::resize(frame, img, cv::Size(64, 64));
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);
    imgTensor = imgTensor.permute({2,0,1});
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    imgTensor = imgTensor.unsqueeze(0);  

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(imgTensor);
    auto output = module.forward(inputs).toTensor();
    int class_id = torch::argmax(output.data()).item<int>();
    return class_id;
} 

void draw_traffic_light(torch::jit::script::Module model,cv::Mat& frame,std::vector<torch::Tensor> dets, std::vector<std::string> classnames)
{
    if (dets.size() > 0)
    {
        for (size_t i=0; i < dets[0].sizes()[0]; ++ i)
        {
            float left = dets[0][i][0].item().toFloat() * frame.cols / 640;
            float top = dets[0][i][1].item().toFloat() * frame.rows / 384;
            float right = dets[0][i][2].item().toFloat() * frame.cols / 640;
            float bottom = dets[0][i][3].item().toFloat() * frame.rows / 384;
            float score = dets[0][i][4].item().toFloat();
            int classID = dets[0][i][5].item().toInt();
            if(classID == 9)
            {
                cv::Rect bbox(left, top, (right - left), (bottom - top));
                cv::Mat cropped_frame = frame(bbox);
                
                int light_classID = classify_traffic_light(model,cropped_frame);
                std::string light_name = "red";
                cv::Scalar color(255, 0, 0);
                
                if(light_classID == 0){} //Red
                else if(light_classID == 1)
                {
                    light_name = "green";
                    color = cv::Scalar(0,255,0);
                } //Green
                
                cv::rectangle(frame, bbox, color, 1);
                cv::putText(frame,
                    light_name + ": " + cv::format("%.2f", score),
                    cv::Point(left, top),
                    cv::FONT_HERSHEY_SIMPLEX, (right - left) / 20, color, 2);

            }
            
        }
    }
}

std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5)
{
        std::vector<torch::Tensor> output;
        for (size_t i=0; i < preds.sizes()[0]; ++i)
        {
            torch::Tensor pred = preds.select(0, i);
            
            // Filter by scores
            torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
            pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
            if (pred.sizes()[0] == 0) continue;

            // (center_x, center_y, w, h) to (left, top, right, bottom)
            pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
            pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
            pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
            pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

            // Computing scores and classes
            std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
            pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
            pred.select(1, 5) = std::get<1>(max_tuple);

            torch::Tensor  dets = pred.slice(1, 0, 6);

            torch::Tensor keep = torch::empty({dets.sizes()[0]});
            torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
            std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
            torch::Tensor v = std::get<0>(indexes_tuple);
            torch::Tensor indexes = std::get<1>(indexes_tuple);
            int count = 0;
            while (indexes.sizes()[0] > 0)
            {
                keep[count] = (indexes[0].item().toInt());
                count += 1;

                // Computing overlaps
                torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
                for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
                {
                    lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                    tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                    rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                    bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                    widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                    heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
                }
                torch::Tensor overlaps = widths * heights;

                // FIlter by IOUs
                torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
                indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
            }
            keep = keep.toType(torch::kInt64);
            output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
        }
        return output;
}