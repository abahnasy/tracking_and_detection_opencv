

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <random> // For std::mt19937, std::random_device

#include "DataContainers.h" // Prediction data type for Random Forest which provides predicted label, bounding box and prediction confidence.

class RandomForest
{
public:
	RandomForest();

    // You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);
    
    ~RandomForest();

    void setTreeCount(int treeCount);
    void setMaxDepth(int maxDepth);
    void setCVFolds(int cvFols);
    void setMinSampleCount(int minSampleCount);
    void setMaxCategories(int maxCategories);
    void train(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,float subsetPercentage, bool data_augmentaion = false);
    DetectedObject predict(cv::Mat &testImage);
private:
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;
    // M-Trees for constructing the forest
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;

    std::mt19937 m_randomGenerator;

    cv::Size winSize_;
    cv::HOGDescriptor hog_detector_;
	cv::Size winStride_;
	cv::Size padding_;


    std::vector<int> getRandomUniqueIndices(int start, int end, int numOfSamples);
    std::vector<std::pair<int, cv::Mat>> generateTrainingImagesLabelSubsetVector(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector, float subsetPercentage);
    std::vector<cv::Mat> augmentImage(cv::Mat &image);
};

#endif //RF_RANDOMFOREST_H
