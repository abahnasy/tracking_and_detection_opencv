/*
 * LoadDataset_Task2.cpp
 *
 *  Created on: Jan 3, 2020
 *      Author: abahnasy
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>


std::vector<std::vector<std::pair<int, cv::Mat>>> load_dataset_task2 (void) {

	std::vector<std::pair<int, cv::Mat>> train_images;
	std::vector<std::pair<int, cv::Mat>> test_images;

	std::vector<int> train_images_num {49, 67, 42, 53, 67, 110};
	std::vector<int> test_images_num {10, 10, 10, 10, 10, 10};

	train_images.reserve(314); // total number of training examples = 49 + 67 + 42 + 53 + 67 + 110 = 314
	test_images.reserve (60); // total number of test images

	for (int i = 0; i < train_images_num.size(); i++) {
		std::cout << i << std::endl;

		for (int iter_train = 0; iter_train < train_images_num.at(i); ++iter_train) {
			// formulate the name of the image to be read
			std::stringstream image_path;
			image_path << "/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task2/train" << "/" << std::setfill('0') << std::setw(2) << i <<"/" << std::setfill('0') << std::setw(4) << iter_train << ".jpg";
			// convert it to string
			std::string image_path_str = image_path.str();
			std::pair<int, cv::Mat> temp_container;
			temp_container.first = i;
			temp_container.second = cv::imread(image_path_str, cv::IMREAD_UNCHANGED);
			train_images.push_back(temp_container);
		}

		for (int iter_test = 0; iter_test < test_images_num.at(i); ++iter_test) {
			// formulate the name of the image to be read
			std::stringstream image_path;
			image_path << "/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task2/test" << "/" << std::setfill('0') << std::setw(2) << i <<"/" << std::setfill('0') << std::setw(4) << train_images_num.at(i) + iter_test << ".jpg";
			// convert it to string
			std::string image_path_str = image_path.str();
			std::pair<int, cv::Mat> temp_container;
			temp_container.first = i;
			temp_container.second = cv::imread(image_path_str, cv::IMREAD_UNCHANGED);
			test_images.push_back(temp_container);
		}

	}

	std::vector<std::vector<std::pair<int, cv::Mat>>> full_dataset;
	// add test and train dataset in a combined structure
	full_dataset.push_back(train_images);
	full_dataset.push_back(test_images);
	return full_dataset;
}


