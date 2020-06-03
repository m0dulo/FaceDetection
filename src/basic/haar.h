#ifndef FACEDETECTION_HAAR_H
#define FACEDETECTION_HAAR_H

#include "detector.h"
#include <vector>

class HaarDetector : public Detector {
private:
  cv::CascadeClassifier _faceDetector;
  std::string _xmlPath;

public:
  HaarDetector(std::string imgPath, std::string videoPath)
      : Detector(imgPath, videoPath) {}

  void init(const std::string modelPath) override {
    this->_xmlPath = modelPath;
  }

  void imgDetect() override {
    if (!_faceDetector.load(_xmlPath)) {
      LyxUtilis::log("无法加载模型文件！");
      return;
    }
    std::vector<cv::Rect> detectRes;
    cv::Mat m = cv::imread(_imgPath);
    cv::Mat temp = m.clone();
    cv::cvtColor(m, temp, cv::COLOR_BGR2GRAY);
    _faceDetector.detectMultiScale(temp, detectRes, 1.1, 3);
    for (auto &res : detectRes) {
      cv::rectangle(m, res, cv::Scalar(25, 0, 0), 2);
    }
    cv::imshow("Haar Img Detection Res", m);
    cv::waitKey(3000);
  }

  void videoDetect() override {
    cv::VideoCapture v(_videoPath);
    cv::Mat frame, temp;
    while (true) {
      std::vector<cv::Rect> detectRes;
      if (!v.read(frame)) {
        LyxUtilis::log("播放结束！");
        return;
      }
      cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
      _faceDetector.detectMultiScale(temp, detectRes, 1.1, 3);
      for (auto &res : detectRes) {
        cv::rectangle(frame, res, cv::Scalar(25, 0, 0), 2);
      }
      cv::imshow("Haar Video Detection Res", frame);
      cv::waitKey(v.get(cv::CAP_PROP_FPS));
    }
  }
};
#endif // FACEDETECTION_HAAR_H
