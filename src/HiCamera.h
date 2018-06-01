#ifndef HI_CAPTURE_H
#define HI_CAPTURE_H

#include <dlib/matrix.h>
#include "Hi.h"

using namespace dlib;

class HiCamera {

public:
    /**
     * Capture numFrames using device captureDeviceID
     *
     * @param captureDeviceID
     * @param numFrames
     * @return std::vector<matrix<rgb_pixel>>
     */
    static std::vector<matrix<rgb_pixel>> capture(int captureDeviceID = 0, int numFrames = 15);

    /**
     * Capture up to maxFrames using device captureDeviceID, untill a face is found
     *
     * @param captureDeviceID
     * @param maxFrames
     * @return std::vector<matrix<rgb_pixel>>
     */
    static matrix<rgb_pixel> captureFace(Hi &hi, int captureDeviceID = 0, int maxFrames = 30);
};


#endif //HI_CAPTURE_H
