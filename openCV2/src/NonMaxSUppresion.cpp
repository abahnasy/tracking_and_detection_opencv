/*
 * NonMaxSUppresion.cpp
 *
 *  Created on: Jan 8, 2020
 *      Author: bahnasya
 */
#include "DataContainers.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

#  define DEBUG(x) std::cout << x << std::endl

// Compares two intervals according to staring times.
bool compareInterval(DetectedObject i1, DetectedObject i2)
{
    return (i1.confidence < i2.confidence);
}

bool checkLabel(DetectedObject i1, DetectedObject i2) {
	return (i1.label == i2.label);
}

std::vector<DetectedObject> non_maximum_suppresion(cv::Mat image, std::vector<DetectedObject> proposals, float NMS_MAX_IOU_THRESHOLD) {

  std::sort(proposals.begin(), proposals.end(), compareInterval);

  std::vector<DetectedObject> nms_filtered_proposals;



  while(proposals.size() != 0) {
    nms_filtered_proposals.push_back(proposals.back());
    proposals.pop_back();
    std::vector<DetectedObject>::iterator iter;
    DetectedObject pick_last_from_nms_filtered = nms_filtered_proposals.back();
//    int debug = 0;
    for (iter = proposals.begin(); iter != proposals.end(); ) {
//    	DEBUG(debug++);
    	/* Extract  the bounding box to  compare */
		cv::Rect &rect1 =iter->bounding_box;
		cv::Rect &rect2 = pick_last_from_nms_filtered.bounding_box;
		float iouScore = ((rect1 & rect2).area() * 1.0f) / ((rect1 | rect2).area());

		if (iouScore > NMS_MAX_IOU_THRESHOLD && checkLabel(pick_last_from_nms_filtered, *iter)) // Merge the two bounding boxes
		{
//			DEBUG("Delete happens");
			iter = proposals.erase(iter);
		} else {
			++iter;
		}
    }



	}

  return nms_filtered_proposals;
}



