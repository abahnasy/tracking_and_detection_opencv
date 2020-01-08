/*
 * task3.cpp
 *
 *  Created on: Jan 6, 2020
 *      Author: abahnasy
 */

#include <opencv2/opencv.hpp>
#include "RandomForest.h"
#include "DataContainers.h"

#include "ComputingLocation.h"

#ifdef RECHNERHALLE
// C++ program to create a directory in Linux
#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <opencv2/core/utils/filesystem.hpp>
#endif


#  define DEBUG(x) std::cout << x << std::endl

void load_dataset_task3 (
		std::vector<std::vector<std::pair<int, cv::Mat>>>& ,
		std::vector<std::vector<std::vector<int>>>&
		);


std::vector<float> compute_metrics(std::vector<DetectedObject> predictionsNMSVector,
                            std::vector<DetectedObject> groundTruthPredictions)
{
    float tp = 0, fp = 0, fn = 0;
    float matchThresholdIou = 0.5f;

    for (auto &&myPrediction : predictionsNMSVector)
    {
        bool matchesWithAnyGroundTruth = false;
        cv::Rect myRect = myPrediction.bounding_box;

        for (auto &&groundTruth : groundTruthPredictions)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            cv::Rect gtRect = groundTruth.bounding_box;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                matchesWithAnyGroundTruth = true;
                break;
            }
        }
        if (matchesWithAnyGroundTruth)
            tp++;
        else
            fp++;
    }

    for (auto &&groundTruth : groundTruthPredictions)
    {
        bool isGtBboxMissed = true;
        cv::Rect gtRect = groundTruth.bounding_box;
        for (auto &&myPrediction : predictionsNMSVector)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            cv::Rect myRect = myPrediction.bounding_box;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                isGtBboxMissed = false;
                break;
            }
        }

        if (isGtBboxMissed)
            fn++;
    }

    std::vector<float> results;
    results.push_back(tp);
    results.push_back(fp);
    results.push_back(fn);
    return results;
}


std::vector<float> task3(float confidence_threshold) {

	// load train dataset
	// create random forest for 4 classes
	// train the random forest using the train data

	// load test dataset
	// run sliding window for every image and extract vector of predictions
	// loop over vector of predictions to extract the bounding boxes text values and write them in a log file
	// print image with all bounding boxes in output folder

	float NMS_CONFIDENCE_THRESHOLD = confidence_threshold;
	float NMS_MIN_IOU_THRESHOLD = 0.1f;
	float NMS_MAX_IOU_THRESHOLD = 0.5f;

	std::vector<std::vector<std::pair<int, cv::Mat>>> full_dataset;

	std::vector<std::vector<std::vector<int>>> groundTruthBoundingBoxes;

	load_dataset_task3(full_dataset,groundTruthBoundingBoxes);

    cv::Scalar bounding_box_colors[4];
    bounding_box_colors[0] = cv::Scalar(255, 0, 0);
    bounding_box_colors[1] = cv::Scalar(0, 255, 0);
    bounding_box_colors[2] = cv::Scalar(0, 0, 255);
    bounding_box_colors[3] = cv::Scalar(255, 255, 0);

	// load the training part from the dataset
	std::vector<std::pair<int, cv::Mat>> trainingImagesLabelVector = full_dataset.at(0);

	//create random forest
	int treeCount = 40;
	int maxDepth = 5;
	int CVFolds = 1; // Not implemented Error, set to 1
	int minSampleCount = 2;
	int maxCategories = 4;
	bool data_augmentation = false;

	float subsetPercentage = 80.0f;
	RandomForest *rf = new RandomForest(treeCount,  maxDepth,  CVFolds,  minSampleCount,  maxCategories);
//	train random forest
	rf->train(trainingImagesLabelVector,subsetPercentage, data_augmentation);


	/*Run sliding window with different scales over every test image and get the result of prediction for every window */

	// load test dataset
	std::vector<std::pair<int, cv::Mat>> testImagesLabelVector = full_dataset.at(1);
	// define stride values;
	int rows_stride = 8;
	int cols_stride = 8;

	// open a stream file
	std::ostringstream s;

#ifdef RECHNERHALLE
	s << "data/task3/output/Trees-" << treeCount << "confidence_threshold" << NMS_CONFIDENCE_THRESHOLD << "-augment_" << data_augmentation << "-strideX_" << cols_stride << "-strideY_" << rows_stride << "/";;
	std::string outputDir = s.str();
	const char * c = outputDir.c_str();
	if (mkdir(c, 0777) == -1)
	        std::cerr << "Error :  " << strerror(errno) << std::endl;

	    else
	    	std::cout << "Directory created";
#else
	s << "/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task3/output/Trees-" << treeCount << "confidence_threshold" << NMS_CONFIDENCE_THRESHOLD << "-augment_" << data_augmentation << "-strideX_" << cols_stride << "-strideY_" << rows_stride << "/";
	std::string outputDir = s.str();
	cv::utils::fs::createDirectory(outputDir);
#endif






	std::ofstream predictionsFile(outputDir + "_predictions.txt");
	if (!predictionsFile.is_open())
	{
		std::cout << "Failed to open" << outputDir + "_predictions.txt" << std::endl;
		exit(-1);
	}

	// metrics used to asses the overall performance across all images.
	float tp = 0, fp = 0, fn = 0;
	for(int i = 0; i <testImagesLabelVector.size(); ++i) {
		std::cout << "Running sliding window on image # " << i << " of " << testImagesLabelVector.size() << "\n";
		predictionsFile << i << std::endl; // Prediction file format: Starts with File number

		cv::Mat test_image = testImagesLabelVector.at(i).second;
		std::cout << "Test Image Dimensions " << test_image.rows << " , " << test_image.cols << "\n";
		std::vector<DetectedObject> predictions_per_image;
		// define the sliding window size
		int sliding_window_length = 64;
		int max_slide_window = std::max(test_image.cols, test_image.rows);
		int window_scale_factor = 2;


		while(sliding_window_length < max_slide_window) {
			cv::Mat rec_img = test_image.clone();
			for(int rows = 0; rows < (test_image.rows - sliding_window_length); rows += rows_stride) {
				for(int cols = 0; cols < (test_image.cols - sliding_window_length); cols += cols_stride) {

					cv::Rect r (cols,rows, sliding_window_length, sliding_window_length);

					cv::Mat test_image_window_field_of_view = test_image(r);

					/* DEBUG */
					//cv::rectangle(rec_img, r, cv::Scalar(255), 1, 8, 0);
					//cv::imshow("Single sliding window", rec_img);
					//cv::waitKey(10);

					// Predict on sliding window field of view
					DetectedObject prediction = rf->predict(test_image_window_field_of_view);
					if (prediction.label != 3) // Ignore Background class.
					{
						prediction.bounding_box = r;
						predictions_per_image.push_back(prediction);
					}

				}
			}
			sliding_window_length *= window_scale_factor;
		}
		// Prediction file format: Next is N Lines of Labels, cv::Rect and confidence
		predictionsFile << predictions_per_image.size() << std::endl;
		for (auto &&prediction : predictions_per_image)
		{
			// Prediction file format: Next is N Lines of Labels and cv::Rect
			predictionsFile << prediction.label << " " << prediction.bounding_box.x << " " << prediction.bounding_box.y << " " << prediction.bounding_box.height << " " << prediction.bounding_box.width << " " << prediction.confidence << std::endl;
		}
		cv::Mat testImageClone = test_image.clone(); // For drawing bbox
		for (auto &&prediction : predictions_per_image) {
			cv::rectangle(testImageClone, prediction.bounding_box, bounding_box_colors[prediction.label]);
		}


		// Draw bounding box on the test image using ground truth
		std::vector<std::vector<int>> imageLabelsAndBoundingBoxes = groundTruthBoundingBoxes.at(i);
		//imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
		cv::Mat testImageGtClone = test_image.clone(); // For drawing bbox
		for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
		{
			std::vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
			cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
			cv::rectangle(testImageGtClone, rect, bounding_box_colors[bbox[0]]);
		}

		std::stringstream modelOutputFilePath;
		modelOutputFilePath << outputDir << std::setfill('0') << std::setw(4) << i << "-ModelOutput.png";
		std::string modelOutputFilePathStr = modelOutputFilePath.str();
		cv::imwrite(modelOutputFilePathStr, testImageClone);

		std::stringstream gtFilePath;
		gtFilePath << outputDir << std::setfill('0') << std::setw(4) << i << "-GrountTruth.png";
		std::string gtFilePathStr = gtFilePath.str();
		cv::imwrite(gtFilePathStr, testImageGtClone);





		// Run NMS over the vector of predicted bounding boxes
		cv::Mat testImageNmsClone = test_image.clone();
		std::vector<DetectedObject> predictions_per_image_NMS;
		predictions_per_image_NMS.reserve(25);

		// erase boxes with confidence below the threshold
		std::cout << "Erase the predictions which have confidence below the threshold value for image: " << i << "\n";
		std::cout <<"Total predictions before thresholding are " << predictions_per_image.size() << "\n";
		std::vector<DetectedObject>::iterator iter;
		for (iter = predictions_per_image.begin(); iter != predictions_per_image.end(); ) {
			if (iter->confidence < NMS_CONFIDENCE_THRESHOLD)
				iter = predictions_per_image.erase(iter);
			else
				++iter;
		}
		std::cout <<"Total predictions after thresholding are " << predictions_per_image.size() << "\n";
		std::cout << "Calculating the IOU value for remaining vectors" << "\n";
		for (auto &&prediction : predictions_per_image){
			bool similar_prediction_flag = false;
			for (auto &&predicion_NMS : predictions_per_image_NMS)
			{
				if (predicion_NMS.label == prediction.label)
				{ // Only if same label
					/* Extract  the bounding box to  compare */
					cv::Rect &rect1 = prediction.bounding_box;
					cv::Rect &rect2 = predicion_NMS.bounding_box;
					float iouScore = ((rect1 & rect2).area() * 1.0f) / ((rect1 | rect2).area());
					//std::cout << "iou score is "  << iouScore << "\n";
					if (iouScore > NMS_MAX_IOU_THRESHOLD) // Merge the two bounding boxes
					{
						//std::cout <<"iou score passed the threshold check \n";
						predicion_NMS.bounding_box = rect1 | rect2;
						predicion_NMS.confidence = std::max(prediction.confidence, predicion_NMS.confidence);
						similar_prediction_flag = true;
						break;
					}
					 else if (iouScore > NMS_MIN_IOU_THRESHOLD) // ToDo: Improve this.
					 {
					     // Drop the bounding box with lower confidence
					     if (predicion_NMS.confidence < prediction.confidence)
					     {
					    	 predicion_NMS = prediction;
					     }
					     similar_prediction_flag = true;
					     break;
					 } else {
						 // similar but iou score is below minimum
					 }
				}
			}
			// If no NMS cluster found, add the prediction as a new cluster
			if (!similar_prediction_flag)
				predictions_per_image_NMS.push_back(prediction);
		}
		// Prediction file format: Next is N Lines of Labels and cv::Rect
		for (auto &&predicion_NMS : predictions_per_image_NMS)
			cv::rectangle(testImageNmsClone, predicion_NMS.bounding_box, bounding_box_colors[predicion_NMS.label]);

        // Write NMS output image
		std::cout <<"Writing the image after processing NMS algorithm" << "\n";
        std::stringstream nmsOutputFilePath;
        nmsOutputFilePath << outputDir << std::setfill('0') << std::setw(4) << i << "-NMSOutput" << "-Confidence-" << NMS_CONFIDENCE_THRESHOLD << ".png";
        std::string nmsOutputFilePathStr = nmsOutputFilePath.str();
        cv::imwrite(nmsOutputFilePathStr, testImageNmsClone);


        std::vector<DetectedObject> groundTruthPredictions;
		for (size_t j = 0; j < groundTruthBoundingBoxes.at(i).size(); j++)
		{
			DetectedObject ground_truth_predictions;
			ground_truth_predictions.label = groundTruthBoundingBoxes.at(i).at(j).at(0);
			ground_truth_predictions.bounding_box.x = groundTruthBoundingBoxes.at(i).at(j).at(1);
			ground_truth_predictions.bounding_box.y = groundTruthBoundingBoxes.at(i).at(j).at(2);
			ground_truth_predictions.bounding_box.height = groundTruthBoundingBoxes.at(i).at(j).at(3);
			ground_truth_predictions.bounding_box.height -= ground_truth_predictions.bounding_box.x;
			ground_truth_predictions.bounding_box.width = groundTruthBoundingBoxes.at(i).at(j).at(4);
			ground_truth_predictions.bounding_box.width -= ground_truth_predictions.bounding_box.y;
			groundTruthPredictions.push_back(ground_truth_predictions);
		}

		// function  used to  calculate PR values
		std::vector<float> metrics = compute_metrics(predictions_per_image_NMS, groundTruthPredictions);
		tp += metrics[0];
		fp += metrics[1];
		fn += metrics[2];
	}
	predictionsFile.close();

	float precision = tp / (tp + fp);
	float recall = tp / (tp + fn);
	std::cout << "NMS_CONFIDENCE_THRESHOLD: " << NMS_CONFIDENCE_THRESHOLD << ", Precision: " << precision << ", Recall: " << recall << "\n";
	std::vector<float> return_vector;
	return_vector.push_back(precision);
	return_vector.push_back(recall);

	return return_vector;










}



