

#ifndef HOGVISUALIZATION_H
#define HOGVISUALIZATION_H

#include <opencv2/opencv.hpp>
using namespace cv;


void visualizeHOG(cv::Mat img,
                  std::vector<float> &feats,
                  cv::HOGDescriptor& hog_detector,
                  int scale_factor = 3);

#endif //HOGVISUALIZATION_H
