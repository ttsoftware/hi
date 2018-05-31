#ifndef HI_CAPTURE_H
#define HI_CAPTURE_H

#include <dlib/matrix.h>

using namespace dlib;

class HiCamera {

public:
    static std::vector<matrix<rgb_pixel>> capture(int captureDeviceID = 0, int numFrames = 15);
};


#endif //HI_CAPTURE_H
