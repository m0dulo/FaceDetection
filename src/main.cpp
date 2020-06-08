#include <memory>
#include "Argument_helper.h"
#include "dnn.h"
#include "haar.h"
#include <opencv2/opencv.hpp>

const std::string haarXmlPath = "../models/haarcascade_frontalface_default.xml";
const std::string dnnModelConfig = "../models/deploy.prototxt";
const std::string dnnModelBinary =
    "../models/res10_300x300_ssd_iter_140000.caffemodel";

int main(int argc, char *argv[]) {
  std::string model = "";
  std::string imgFile = "";
  std::string videoFile = "";

  dsr::Argument_helper ah;
  ah.new_named_string("model", "model_name", "named_string", "named_string",
                      model);
  ah.new_named_string("img", "img_name", "named_string", "named_string",
                      imgFile);
  ah.new_named_string("video", "video_name", "named_string", "named_string",
                      videoFile);
  ah.process(argc, argv);

  Detector *detector;
  if (model == "haar") {
    std::unique_ptr<Detector> detectorPtr(new HaarDetector(imgFile, videoFile));
    detectorPtr->init(haarXmlPath);
    detectorPtr->imgDetect();
    detectorPtr->videoDetect();
  } else if (model == "dnn") {
    std::unique_ptr<Detector> detectorPtr( new DnnDetector(imgFile, videoFile));
    detectorPtr->init(dnnModelConfig, dnnModelBinary);
    detectorPtr->imgDetect();
    detectorPtr->videoDetect();
  }

  return 0;
}
