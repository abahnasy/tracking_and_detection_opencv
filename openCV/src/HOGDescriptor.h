

#ifndef RF_HOGDESCRIPTOR_H
#define RF_HOGDESCRIPTOR_H


#include <opencv2/opencv.hpp>
#include <vector>

class HOGDescriptor {
	private:
		cv::Size win_size;

		/*
			Fill other parameters here
		*/

		cv::HOGDescriptor hog_detector;

		bool is_init;

	public:

		HOGDescriptor() {
			//initialize default parameters(win_size, block_size, block_step,....)
			win_size = cv::Size(64, 64);

			//Fill other parameters here

			// parameter to check if descriptor is already initialized or not
			is_init = false;
		};


		void setWinSize() {
			//Fill

		}

		cv::Size getWinSize(){
			//Fill
		}

		void setBlockSize() {
			//Fill
		}

		void setBlockStep() {
		   //Fill
		}

		void setCellSize() {
		  //Fill
		}

		void setPadSize(cv::Size sz) {
			pad_size = sz;
		}

		cv::HOGDescriptor & getHog_detector();

		void initDetector();

		void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor = 3);

		void detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat, cv::Size sz, bool show);

		~HOGDescriptor() {};



};

#endif //RF_HOGDESCRIPTOR_H
