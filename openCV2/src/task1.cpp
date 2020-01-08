/*
 * task1.cpp
 *
 *  Created on: Jan 2, 2020
 *      Author: abahnasy
 */



#include <opencv2/opencv.hpp>
#include "HogVisualization.h"
#include <iomanip>
#include <sstream>
#include <random>
#include "ComputingLocation.h"


using namespace cv;
using namespace std;

HOGDescriptor createHogDescriptor(Size winSize)
{
    // Create Hog Descriptor
    Size blockSize(16, 16);
    Size blockStride(8, 8);
    Size cellSize(8, 8);
    int nbins(9);
    int derivAperture(1);
    double winSigma(-1);
    int histogramNormType(HOGDescriptor::L2Hys);
    double L2HysThreshold(0.2);
    bool gammaCorrection(true);
    float free_coef(-1.f);
    //! Maximum number of detection window increases. Default value is 64
    int nlevels(HOGDescriptor::DEFAULT_NLEVELS);
    //! Indicates signed gradient will be used or not
    bool signedGradient(false);
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                      L2HysThreshold, gammaCorrection, nlevels, signedGradient);
    return hog;
}

cv::Mat resizeToBoundingBox(cv::Mat &inputImage, Size &winSize)
{
    cv::Mat resizedInputImage;
    if (inputImage.rows < winSize.height || inputImage.cols < winSize.width)
    {
        float scaleFactor = fmax((winSize.height * 1.0f) / inputImage.rows, (winSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedInputImage, Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedInputImage = inputImage;
    }

    Rect r = Rect((resizedInputImage.cols - winSize.width) / 2, (resizedInputImage.rows - winSize.height) / 2,
                  winSize.width, winSize.height);

    // Debugging Code
//    cv::Mat resizedInputImageOverlayed;
//    resizedInputImage.copyTo(resizedInputImageOverlayed);
//    cv::rectangle(resizedInputImageOverlayed, r, cv::Scalar(0, 255, 0));
//    cv::putText(resizedInputImageOverlayed, "Resized",cv::Point(0,20) ,FONT_HERSHEY_SIMPLEX, 1, Scalar(0,200,200), 2);
//    imshow("The selected region from resized image", resizedInputImageOverlayed);

    return resizedInputImage(r);
}

void task1()
{
    // Read image and display
#ifdef RECHNERHALLE
	string imagePath = "/u/halle/bahnasya/home_at/Desktop/tracking_and_detection_opencv/openCV2/data/task1/obj1000.jpg";
#else
	string imagePath = "/home/abahnasy/Desktop/tracking_and_detection_opencv/openCV2/data/task1/obj1000.jpg";
#endif

    cout << imagePath << endl;
    Mat inputImage = imread(imagePath, cv::IMREAD_UNCHANGED);
    imshow("task1 - Input Image", inputImage);
    cv::waitKey(200);

    // Resize image if very small while maintaining aspect ratio till its bigger than winSize
    Size winSize(128, 128);
    cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);
    //imshow("task1 - Input Image after resizing", resizedInputImage);
    cv::waitKey(200);

    HOGDescriptor hog = createHogDescriptor(winSize);

    cv::Mat grayImage;
    cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Compute Hog only of center crop of grayscale image
    vector<float> descriptors;
    vector<Point> foundLocations;
    vector<double> weights;
    Size winStride(8, 8);
    Size padding(0, 0);
    hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);
    visualizeHOG(resizedInputImage, descriptors, hog, 5);
}

