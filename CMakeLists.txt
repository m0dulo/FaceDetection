cmake_minimum_required(VERSION 3.0)
project(FaceDetection)
find_package(OpenCV REQUIRED)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
if(CMAKE_BUILD_TYPE MATCHES debug)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -O1" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -O3 -march=native" )
endif()
include_directories(${OpenCV_INCLUDE_DIRS}
                    src/basic
                    src/utilis
                    third_party/Argument_helper)
add_executable(FaceDetection src/main.cpp)
target_link_libraries(FaceDetection ${OpenCV_LIBS})