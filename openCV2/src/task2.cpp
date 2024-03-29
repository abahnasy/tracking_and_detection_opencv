/*
 * task2.cpp
 *
 *  Created on: Jan 2, 2020
 *      Author: abahnasy
 */

#include <opencv2/opencv.hpp>

//#include "HOGDescriptor.h"
#include "RandomForest.h"

#  define DEBUG(x) std::cout << x << std::endl

cv::HOGDescriptor createHogDescriptor(cv::Size winSize);
cv::Mat resizeToBoundingBox(cv::Mat &inputImage, cv::Size &winSize);


using namespace std;

std::vector<std::vector<std::pair<int, cv::Mat>>> load_dataset_task2 (void);

template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> data) {

	/*
		Fill Code
	*/

};





void testDTrees() {

    int num_classes = 6;

    /*
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance
      * Experiment with the MaxDepth parameter, to see how it affects the performance

    */

    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = load_dataset_task2();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);
    // Create the model
	cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
	model->setCVFolds(0); // set num cross validation folds - Not implemented in OpenCV
	model->setMaxCategories(6);  // set max number of categories
	model->setMaxDepth(5);       // set max tree depth
	model->setMinSampleCount(2); // set min sample count
	cout << "Number of cross validation folds are: " << model->getCVFolds() << endl;
	cout << "Max Categories are: " << model->getMaxCategories() << endl;
	cout << "Max depth is: " << model->getMaxDepth() << endl;
	cout << "Minimum Sample Count: " << model->getMinSampleCount() << endl;


	// Compute Hog Features for all the training images
	cv::Size winSize(128, 128);
	cv::HOGDescriptor hog = createHogDescriptor(winSize);
	cv::Size winStride(16, 16);
	cv::Size padding(0, 0);


    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = trainingImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

        // Compute Hog only of center crop of grayscale image
        std::vector<float> descriptors;
        std::vector<cv::Point> foundLocations;
        std::vector<double> weights;
        hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
         cout << "=====================================" << endl;
         cout << "Number of descriptors are: " << descriptors.size() << endl;
         cv::Mat new_vector_after_reshape =  cv::Mat(descriptors).clone().reshape(1, 1);
        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        cout << "Decision Tree Vector" << feats.size() << endl;
         cout << "New size of training features" << new_vector_after_reshape.size() << endl;
        labels.push_back(trainingImagesLabelVector.at(i).first);
//         cout << "New size of training labels" << labels.size() << endl;
    }

	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);


	model->train(trainData);

	// Predict on test dataset
	std::vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
	float accuracy = 0;
//	cv::Size winSize(128, 128);
//	cv::HOGDescriptor hog = createHogDescriptor(winSize);
//	cv::Size winStride(8, 8);
//	cv::Size padding(0, 0);

	for (size_t i = 0; i < testImagesLabelVector.size(); i++)
	{

		cv::Mat inputImage = testImagesLabelVector.at(i).second;
		imshow("task1 - Input Image", inputImage);
		cv::waitKey(200);
		cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

		// Compute Hog only of center crop of gray scale image
		std::vector<float> descriptors;
		std::vector<cv::Point> foundLocations;
		vector<double> weights;
		hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

		// Store the features and labels for model training.
		// cout << i << ": Expected: " << testImagesLabelVector.at(i).first << ", Found: " << model->predict(cv::Mat(descriptors)) << endl ;
		if (testImagesLabelVector.at(i).first == model->predict(cv::Mat(descriptors)))
			accuracy += 1;
	}

	cout << "==================================================" << endl;
	cout << "TASK 2 - Single Decision Tree Accuracy is: [" << accuracy / testImagesLabelVector.size() << "]." << endl;
	cout << "==================================================" << endl;





    //performanceEval<cv::ml::DTrees>(tree, train_data);
    //performanceEval<cv::ml::DTrees>(tree, test_data);

}


void testForest(){



    /*
      *
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */
    vector<vector<pair<int, cv::Mat>>> dataset = load_dataset_task2();
	vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);

	//create random forest
	int treeCount = 60;
	int maxDepth = 30;
	int CVFolds = 1; // Not implemented Error, set to 1
	int minSampleCount = 2;
	int maxCategories = 6;

	float subsetPercentage = 80.0f;
	RandomForest *rf = new RandomForest(treeCount,  maxDepth,  CVFolds,  minSampleCount,  maxCategories);
	//train random forest
	rf->train(trainingImagesLabelVector,subsetPercentage, true);

	vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
    float accuracy = 0;
    float accuracyPerClass[maxCategories] = {0};
    float ground_truth_class [maxCategories] = {0};
    for (uint8_t i = 0; i < testImagesLabelVector.size(); ++i)
    {
        cv::Mat testImage = testImagesLabelVector.at(i).second;
        ground_truth_class[testImagesLabelVector.at(i).first]++;
        DetectedObject prediction = rf->predict(testImage);
        if (testImagesLabelVector.at(i).first == prediction.label)
        {
            accuracy += 1;
            accuracyPerClass[prediction.label] += 1;
        }

    }

    for (int i = 0; i < maxCategories; ++i) {
    	float res = accuracyPerClass[i] / ground_truth_class[i];
    	cout << "The accuracy of class " << i << " is " << res << "\n";
    }
    cout << "==================================================" << endl;
    cout << "Accuracy is " << accuracy/testImagesLabelVector.size() << endl;
    cout << "==================================================" << endl;



	//predict random forest

//	int num_classes = 6;
//	int numberOfDTrees = 40;
//	cv::Size winSize(128, 128);


//    performanceEval<RandomForest>(forest, train_data);
//    performanceEval<RandomForest>(forest, test_data);
}
