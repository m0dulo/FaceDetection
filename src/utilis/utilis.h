#ifndef FACEDETECTION_UTILIS_H
#define FACEDETECTION_UTILIS_H

#include <iostream>

class LyxUtilis {
public:
  template <class T> static void _print(T arg) { std::cout << arg << " "; }

  template <class... Args> static void log(Args... args) {
    int arr[] = {(_print(args), 0)...};
    std::cout << std::endl;
  }
};

#endif // FACEDETECTION_UTILIS_H
