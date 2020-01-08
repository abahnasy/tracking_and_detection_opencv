/*
 * LoadDataset_Task3.cpp
 *
 *  Created on: Jan 6, 2020
 *      Author: abahnasy
 */
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>

#  define DEBUG(x) std::cout << x << std::endl

void load_dataset_task3 (
		std::vector<std::vector<std::pair<int, cv::Mat>>> &full_dataset,
		std::vector<std::vector<std::vector<int>>> &groundTruthBoundingBoxes
		)
{

	std::vector<std::pair<int, cv::Mat>> train_images;
	std::vector<int> train_images_num {53, 81, 51, 290};

	std::vector<std::pair<int, cv::Mat>> test_images;
	std::vector<int> test_images_num {44};

//	std::vector<std::vector<std::vector<int>>> groundTruthBoundingBoxes;



	int train_sum_of_elems = 0;
	for (auto& n : train_images_num)
		train_sum_of_elems += n;

	train_images.reserve(train_sum_of_elems);

	for (int i = 0; i < train_images_num.size(); i++) {
		for (int iter_train = 0; iter_train < train_images_num.at(i); ++iter_train) {
			// formulate the name of the image to be read
			std::stringstream image_path;
			image_path << "/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task3/train" << "/" << std::setfill('0') << std::setw(2) << i <<"/" << std::setfill('0') << std::setw(4) << iter_train << ".jpg";
			// convert it to string
			std::string image_path_str = image_path.str();
			//DEBUG(image_path_str); /* Debugging */
			std::pair<int, cv::Mat> temp_container;
			temp_container.first = i;
			temp_container.second = cv::imread(image_path_str, cv::IMREAD_UNCHANGED);
			train_images.push_back(temp_container);
		}

	}


	int test_sum_of_elems = 0;
	for (auto& n : test_images_num)
		test_sum_of_elems += n;
	test_images.reserve(test_sum_of_elems);

	for (int i = 0; i < test_images_num.size(); i++) {
		for (int iter_test = 0; iter_test < test_images_num.at(i); ++iter_test) {
			// formulate the name of the image to be read
			std::stringstream image_path;
			image_path << "/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task3/test" << "/" << std::setfill('0') << std::setw(4) << iter_test << ".jpg";
			// convert it to string
			std::string image_path_str = image_path.str();
//			DEBUG(image_path_str); /* Debugging */
			std::pair<int, cv::Mat> temp_container;
			// No Classes to be defined
			temp_container.first = -1;
			temp_container.second = cv::imread(image_path_str, cv::IMREAD_UNCHANGED);
			test_images.push_back(temp_container);

			/* Load Ground truth data bounding box coordinates from text files */

			// construct the path of the current image
			std::stringstream gtFilePath;
			gtFilePath << "/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task3/gt/" << std::setfill('0') << std::setw(4) << iter_test << ".gt.txt";
			std::string gtFilePathStr = gtFilePath.str();

	        std::fstream gtFile;
	        gtFile.open(gtFilePathStr);
	        if (!gtFile.is_open())
	        {
	        	std::cout << "Failed to open file: " << gtFilePathStr << std::endl;
	            exit(-1);
	        }
	        std::string line;
			std::vector<std::vector<int>> groundTruthBoundingBoxesPerImage;
			while (std::getline(gtFile, line))
			{
				std::istringstream in(line);
				// each line contains 5 values
				std::vector<int> groundTruthLabelAndBoundingBox(5);
				int temp;
				for (size_t i = 0; i < 5; i++)
				{
					in >> temp;
					groundTruthLabelAndBoundingBox.at(i) = temp;
				}
				groundTruthBoundingBoxesPerImage.push_back(groundTruthLabelAndBoundingBox);

				//std::cout << groundTruthLabelAndBoundingBox.at(0) << " "<< groundTruthLabelAndBoundingBox.at(1) << " " << groundTruthLabelAndBoundingBox.at(2) << " " << groundTruthLabelAndBoundingBox.at(3) << " " << groundTruthLabelAndBoundingBox.at(4) << " " <<std::endl;
			}
			groundTruthBoundingBoxes.push_back(groundTruthBoundingBoxesPerImage);
		}
	}


	//std::vector<std::vector<std::pair<int, cv::Mat>>> full_dataset;
	// add test and train dataset in a combined structure
	full_dataset.push_back(train_images);
	full_dataset.push_back(test_images);


	//return full_dataset;




}




