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
    detector = new HaarDetector(imgFile, videoFile);
    detector->init(haarXmlPath);
    detector->imgDetect();
    detector->videoDetect();
  } else if (model == "dnn") {
    detector = new DnnDetector(imgFile, videoFile);
    detector->init(dnnModelConfig, dnnModelBinary);
    detector->imgDetect();
    detector->videoDetect();
  }
  delete detector;
  return 0;
}
