/*
 * DataContainers.h
 *
 *  Created on: Jan 6, 2020
 *      Author: abahnasy
 */


#ifndef DATACONTAINERS_H_
#define DATACONTAINERS_H_

#include <opencv2/opencv.hpp>

struct DetectedObject
{
  int label;
  float confidence;
  cv::Rect bbox;
};



#endif /* DATACONTAINERS_H_ */
