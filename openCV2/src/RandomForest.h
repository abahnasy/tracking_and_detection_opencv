

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <random> // For std::mt19937, std::random_device

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
	

    void train(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,float subsetPercentage);

    int predict(cv::Mat &testImage);


private:
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;
    // M-Trees for constructing thr forest
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;

    std::mt19937 m_randomGenerator;

    cv::Size winSize_;
    cv::HOGDescriptor hog_detector_;
	cv::Size winStride_;
	cv::Size padding_;


    std::vector<int> getRandomUniqueIndices(int start, int end, int numOfSamples);
    std::vector<std::pair<int, cv::Mat>> generateTrainingImagesLabelSubsetVector(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector, float subsetPercentage);
};

#endif //RF_RANDOMFOREST_H
