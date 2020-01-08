



#include <opencv2/opencv.hpp>

#include "ComputingLocation.h"



// change the name of the macro definition to run the desired task
#define TASK_3

// Task 1 function headers
void task1();
// Task 2 function headers
void testDTrees();
void testForest();

// Task 3
std::vector<float> task3(float);

int main() {
    // // Task 1
#ifdef TASK_1
     task1();
     cv::waitKey(0);
     cv::destroyAllWindows();
#endif

     // Task 2
#ifdef TASK_2
//     testDTrees();
     testForest();
#endif

// Task 3
//   task3();
//    cv::destroyAllWindows();

#ifdef TASK_3
	std::ofstream outputFile;
#ifdef RECHNERHALLE
	outputFile.open("/u/halle/bahnasya/home_at/Desktop/tracking_and_detection_opencv/openCV2/data/task3/output/predictionRecallValues.csv");
#else

	outputFile.open("/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task3/output/predictionRecallValues.csv");
#endif



	if (!outputFile.is_open())
	{
	   std::cout << "Failed to open" << "predictionRecallValues.csv" << std::endl;
	   exit(-1);
	}
	outputFile << "Precision,Recall"<< std::endl;
	for (int confidence = 65; confidence <= 66; confidence += 5) // If float is used, it may overshoot 1.0 - floating point error
	{
	   std::cout << "Calculating with confidence threshold " << confidence << std::endl;
	   float NMS_CONFIDENCE_THRESHOLD = confidence / 100.0f;
	   std::vector<float> precisionRecallValue = task3(NMS_CONFIDENCE_THRESHOLD);
	   //std::cout << "NMS_CONFIDENCE_THRESHOLD: " << NMS_CONFIDENCE_THRESHOLD << ", Precision: " << precisionRecallValue[0] << ", Recall: " << precisionRecallValue[1] << std::endl;
	   outputFile << precisionRecallValue[0] << "," << precisionRecallValue[1] << std::endl;
	}
	outputFile.close();
#endif

    return 0;
}
