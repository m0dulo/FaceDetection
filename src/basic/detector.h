#ifndef FACEDETECTION_DETECTOR_H
#define FACEDETECTION_DETECTOR_H

#include "utilis.h"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class Detector {
protected:
  std::string _imgPath;
  std::string _videoPath;

public:
  Detector(const std::string imgPath, const std::string videoPath)
      : _imgPath(imgPath), _videoPath(videoPath) {}
  virtual void init(const std::string xmlPath){};
  virtual void init(const std::string modelConfig,
                    const std::string modelBinary){};
  virtual void imgDetect(){};
  virtual void videoDetect(){};
};

#endif // FACEDETECTION_DETECTOR_H
