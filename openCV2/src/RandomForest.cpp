#include "RandomForest.h"

// external definitions
cv::HOGDescriptor createHogDescriptor(cv::Size winSize);
cv::Mat resizeToBoundingBox(cv::Mat &inputImage, cv::Size &winSize);


RandomForest::RandomForest()
{
	mTreeCount = 0;
	mMaxDepth = 0;
	mCVFolds = 0;
	mMinSampleCount = 0;
	mMaxCategories = 0;
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
    :mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories)
{
   /*
     construct a forest with given number of trees and initialize all the trees with the
     given parameters
   */

	mTrees.reserve(mTreeCount);

	// Generation of random device
    long unsigned int timestamp = static_cast<long unsigned int>(time(0));
    std::cout << timestamp << std::endl;
    m_randomGenerator = std::mt19937(timestamp);

    // initialize the detector
    winSize_ = cv::Size(128, 128);
    cv::Size winSize(128, 128);
    hog_detector_ = createHogDescriptor(winSize);
    // default values for parameters used for compute function
    winStride_ = cv::Size(8,8);
	padding_ = cv::Size(0,0);

	for (uint8_t i = 0; i < this->mTreeCount; ++i) {
		cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
		model->setCVFolds(this->mCVFolds); // set num cross validation folds - Not implemented in OpenCV
		model->setMaxCategories(this->mMaxCategories);  // set max number of categories
		model->setMaxDepth(this->mMaxDepth);       // set max tree depth
		model->setMinSampleCount(this->mMinSampleCount);
		this->mTrees.push_back(model);
	}

}

RandomForest::~RandomForest()
{
}

void RandomForest::setTreeCount(int treeCount)
{
    this->mTreeCount = treeCount;

}

void RandomForest::setMaxDepth(int maxDepth)
{
    mMaxDepth = maxDepth;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFols)
{
    this->mCVFolds = cvFols;

}

void RandomForest::setMinSampleCount(int minSampleCount)
{
    this->mMinSampleCount = minSampleCount;
}

void RandomForest::setMaxCategories(int maxCategories)
{
    this->mMaxCategories = maxCategories;
}



void RandomForest::train(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,float subsetPercentage, bool data_augmentation)
{
	// Compute Hog Features for all the training images
//	cv::Size winSize(128, 128);
//	cv::HOGDescriptor hog = createHogDescriptor(winSize);
//	cv::Size winStride(8, 8);
//	cv::Size padding(0, 0);

    std::vector<std::pair<int, cv::Mat>> augmentedTrainingImagesLabelVector;
    augmentedTrainingImagesLabelVector.reserve(trainingImagesLabelVector.size() * 32);
    if (data_augmentation)
    {
        for(auto&& trainingImagesLabelSample : trainingImagesLabelVector)
        {
            std::vector<cv::Mat> augmentedImages = augmentImage(trainingImagesLabelSample.second);
            for (auto &&augmentedImage : augmentedImages)
            {
                augmentedTrainingImagesLabelVector.push_back(std::pair<int, cv::Mat>(trainingImagesLabelSample.first, augmentedImage));
            }
        }
    } else {
        augmentedTrainingImagesLabelVector = trainingImagesLabelVector;
    }

    std::cout << "The number of received training images is " << trainingImagesLabelVector.size() << "\n";
	std::cout << "Number of generated augmented dataset is " << augmentedTrainingImagesLabelVector.size() << "\n";

	for (uint8_t i = 0; i < this->mTreeCount; ++i) {
    	std::cout << "Training DTree no. " << i+1 << " of " << this->mTreeCount << " .....\n";

    	// generate Random subset of the main dataset
    	std::vector<std::pair<int, cv::Mat>> trainingImagesLabelSubsetVector =
    	            generateTrainingImagesLabelSubsetVector(trainingImagesLabelVector,
    	                                                    subsetPercentage);
    	// compute HoG descriptors for the subset dataset

	    cv::Mat feats, labels;
	    for (size_t i = 0; i < trainingImagesLabelSubsetVector.size(); i++)
	    {
			cv::Mat inputImage = trainingImagesLabelSubsetVector.at(i).second;
			cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, this->winSize_);

			// Compute Hog only of center crop of grayscale image
			std::vector<float> descriptors;
			std::vector<cv::Point> foundLocations;
			std::vector<double> weights;
			hog_detector_.compute(resizedInputImage, descriptors, winStride_, padding_, foundLocations);

			// Store the features and labels for model training.
//			std::cout << "=====================================" << std::endl;
//			std::cout << "Number of descriptors are: " << descriptors.size() << std::endl;
			feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
//			std::cout << "New size of training features" << feats.size() << std::endl;
			labels.push_back(trainingImagesLabelVector.at(i).first);
//			std::cout << "New size of training labels" << labels.size() << std::endl;
	    }






//    	cv::Ptr<cv::ml::DTrees> model = trainDecisionTree(trainingImagesLabelSubsetVector,
//    	                                                          winStride,
//    	                                                          padding);
    	// Create the model
//		cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
//		model->setCVFolds(this->mCVFolds); // set num cross validation folds - Not implemented in OpenCV
//		model->setMaxCategories(this->mMaxCategories);  // set max number of categories
//		model->setMaxDepth(this->mMaxDepth);       // set max tree depth
//		model->setMinSampleCount(this->mMinSampleCount); // set min sample count
//		std::cout << "Number of cross validation folds are: " << model->getCVFolds() << std::endl;
//		std::cout << "Max Categories are: " << model->getMaxCategories() << std::endl;
//		std::cout << "Max depth is: " << model->getMaxDepth() << std::endl;
//		std::cout << "Minimum Sample Count: " << model->getMinSampleCount() << std::endl;


		cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);

		this->mTrees.at(i)->train(trainData);

    //	this->mTrees.push_back(model);
    }
}

DetectedObject RandomForest::predict(cv::Mat &testImage)
{
	cv::Mat resizedInputImage = resizeToBoundingBox(testImage, this->winSize_);
    std::vector<float> descriptors;
    hog_detector_.compute(resizedInputImage, descriptors, winStride_, padding_);
    int labels[this->mMaxCategories] = {0};
    int maxLabelRecord = -1;
    int maxLabel = -1;
    for(auto &&tree : mTrees) {
    	int label = tree->predict(cv::Mat(descriptors));
//    	std::cout<< "Individual tree prediction: " << label << "\n";
    	labels[label]++;
    	if(labels[label] > maxLabelRecord) {
    		maxLabelRecord = labels[label];
    		maxLabel = label;

    	}
    }
    DetectedObject prediction;
    prediction.label = maxLabel;
    prediction.confidence = ((labels[maxLabel] * 1.0f) / this->mTreeCount);
    return prediction;

}

std::vector<int> RandomForest::getRandomUniqueIndices(int start, int end, int numOfSamples)
{
    std::vector<int> indices;
    indices.reserve(end - start);
    for (size_t i = start; i < end; i++)
        indices.push_back(i);

    std::shuffle(indices.begin(), indices.end(), this->m_randomGenerator);
    // copy(indices.begin(), indices.begin() + numOfSamples, std::ostream_iterator<int>(std::cout, ", "));
    // cout << endl;
    return std::vector<int>(indices.begin(), indices.begin() + numOfSamples);
}

std::vector<std::pair<int, cv::Mat>> RandomForest::generateTrainingImagesLabelSubsetVector(
		std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,
        float subsetPercentage)
{
	std::vector<std::pair<int, cv::Mat>> trainingImagesLabelSubsetVector;


    for (uint8_t label = 0; label < mMaxCategories; ++label)
	{
		// Create a subset vector for all the samples with class label.
		std::vector<std::pair<int, cv::Mat>> temp;
		temp.reserve(2000);
		for (auto &&sample : trainingImagesLabelVector)
			if (sample.first == label)
				temp.push_back(sample);
//		std::cout << "Original size: " << temp.size() << "\n";

		// Compute how many samples to choose for each label for random subset.
		int numOfElements= (temp.size() * subsetPercentage) / 100;

		// Filter numOfElements elements from temp and append to trainingImagesLabelSubsetVector
		std::vector<int> randomUniqueIndices = getRandomUniqueIndices(0, temp.size(), numOfElements);



		for (uint8_t j = 0; j < randomUniqueIndices.size(); ++j)
		{
			std::pair<int, cv::Mat> subsetSample = temp.at(randomUniqueIndices.at(j));
			trainingImagesLabelSubsetVector.push_back(subsetSample);
		}

		// Bagging the rest of samples
		int missing = temp.size() - randomUniqueIndices.size();
		// this should work only if subsampling is above 50 percent
		std::vector<int> uniqueFromSelected = getRandomUniqueIndices(0, trainingImagesLabelSubsetVector.size(), missing);

		for (uint8_t j = 0; j < uniqueFromSelected.size(); ++j)
		{
			std::pair<int, cv::Mat> subsetSample = trainingImagesLabelSubsetVector.at(uniqueFromSelected.at(j));
			trainingImagesLabelSubsetVector.push_back(subsetSample);
		}
//
//		std::cout << "unique size: " << trainingImagesLabelSubsetVector.size() << "\n";
	}


	return trainingImagesLabelSubsetVector;
}

std::vector<cv::Mat> RandomForest::augmentImage(cv::Mat &inputImage)
{
    std::vector<cv::Mat> augmentations;
    cv::Mat currentImage = inputImage;
    cv::Mat rotatedImage, flippedImage, scaled_image;
    float scale_factor_up = 0.2f;
    float scale_factor_down = 0.2f;
    for (size_t j = 0; j < 4; j++)
    {
        if (j == 0)
        {
            rotatedImage = currentImage;
        }
        else
        {
            cv::rotate(currentImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
            augmentations.push_back(rotatedImage);
        }

        for (int i = 0; i <= 1; i++)
        {
            cv::flip(rotatedImage, flippedImage, i);
            augmentations.push_back(flippedImage);
            /* Scale with factor 1.2 5 times bigger and five times small */
            std::cout << "Original Image size is " << flippedImage.size() << "\n";
			for(int s = 1; s < 5; ++s) {
				cv::resize(flippedImage, scaled_image, cv::Size(0,0), 1 + (scale_factor_up*s),1 + (scale_factor_up*s), cv::INTER_LINEAR);
//				std::cout << "Scaled Image size with scale factor " << scale_factor_up*s <<" is: " << scaled_image.size() << "\n";
				augmentations.push_back(scaled_image);
//				cv::imshow("Augmented image preview", scaled_image);
//				cv::waitKey(0);
			}
			for(int s = 1; s < 5; ++s) {
				cv::resize(flippedImage, scaled_image, cv::Size(0,0), 1-(scale_factor_down*s),1-(scale_factor_down*s), cv::INTER_LINEAR);
//				std::cout << "Scaled Image size with scale factor " << scale_factor_down/s <<" is: " << scaled_image.size() << "\n";
				augmentations.push_back(scaled_image);
//				cv::imshow("Augmented image preview", scaled_image);
//				cv::waitKey(0);
			}
        }
        currentImage = rotatedImage;
    }

    return augmentations;
}

