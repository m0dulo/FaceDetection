#ifndef FACEDETECTION_DNN_H
#define FACEDETECTION_DNN_H

#include "detector.h"

using namespace cv;

class DnnDetector : public Detector {
private:
  const size_t _width = 300;
  const size_t _height = 300;

  const double _scaleFactor = 1.0;
  const cv::Scalar _meanVal = cv::Scalar(104.0, 117.0, 123.0);
  const float _minConfidence = 0.5;

  std::string _modelConfig;
  std::string _modelBinary;

  cv::dnn::Net _neuralNet;

public:
  DnnDetector(std::string imgPath, std::string videoPath)
      : Detector(imgPath, videoPath) {}

  void init(const std::string modelConfig,
            const std::string modelBinary) override {
    this->_modelConfig = modelConfig;
    this->_modelBinary = modelBinary;
    this->_neuralNet =
        cv::dnn::readNetFromCaffe(this->_modelConfig, this->_modelBinary);
  }

  void imgDetect() override {
    if (_neuralNet.empty()) {
      LyxUtilis::log("Net is Empty!");
      return;
    }
    cv::Mat img = cv::imread(_imgPath);
    cv::Mat inputBolb = cv::dnn::blobFromImage(
        img, _scaleFactor, cv::Size(_width, _height), _meanVal, false, false);
    _neuralNet.setInput(inputBolb, "data");
    cv::Mat detection = _neuralNet.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F,
                         detection.ptr<float>());
    float threshold = _minConfidence;

    for (size_t i = 0; i < detectionMat.rows; ++i) {
      float confidence = detectionMat.at<float>(i, 2);
      if (confidence > threshold) {
        int xLeftBottom =
            static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
        int yLeftBottom =
            static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
        int xRightTop =
            static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
        int yRightTop =
            static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

        cv::Rect rect(xLeftBottom, yLeftBottom, (xRightTop - xLeftBottom),
                      (yRightTop - yLeftBottom));

        cv::rectangle(img, rect, cv::Scalar(255, 0, 0), 2);
      }
    }
    cv::imshow("DNN Img Detection Res", img);
    cv::waitKey(3000);
  }

  void videoDetect() override {
    if (_neuralNet.empty()) {
      LyxUtilis::log("Net is Empty!");
      return;
    }
    cv::VideoCapture cap(_videoPath);
    if (!cap.isOpened()) {
      LyxUtilis::log("Couldn't open!");
      return;
    }
    while (true) {
      cv::Mat frame;
      if (!cap.read(frame)) {
        return;
      }
      cv::Mat inputBolb =
          cv::dnn::blobFromImage(frame, _scaleFactor, cv::Size(_width, _height),
                                 _meanVal, false, false);
      _neuralNet.setInput(inputBolb, "data");
      cv::Mat detection = _neuralNet.forward("detection_out");
      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F,
                           detection.ptr<float>());
      float threshold = _minConfidence;

      for (size_t i = 0; i < detectionMat.rows; ++i) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > threshold) {
          int xLeftBottom =
              static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
          int yLeftBottom =
              static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
          int xRightTop =
              static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
          int yRightTop =
              static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

          cv::Rect rect(xLeftBottom, yLeftBottom, (xRightTop - xLeftBottom),
                        (yRightTop - yLeftBottom));

          cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
        }
      }
      cv::imshow("DNN Video Detection Res", frame);
      waitKey((1000 / cap.get(cv::CAP_PROP_FPS) / 2));
    }
  }
};

#endif // FACEDETECTION_DNN_H