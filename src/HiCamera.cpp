#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <zconf.h>

#include <dlib/gui_widgets.h>
#include <dlib/image_processing/generic_image.h>
#include <dlib/opencv.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_transforms/equalize_histogram.h>

#include "HiCamera.h"

using namespace dlib;
using namespace cv;
using namespace std;

std::vector<matrix<rgb_pixel>> HiCamera::capture(int captureDeviceID, int numFrames) {

    Mat frame;
    matrix<rgb_pixel> img_matrix;
    std::vector<matrix<rgb_pixel>> frames;

    VideoCapture videoCapture(captureDeviceID); // open the default camera
    if (!videoCapture.isOpened()) throw runtime_error("Could not open camera.");

    for (size_t i = 0; i < numFrames; i++) {
        videoCapture >> frame; // get a new frame from camera
        // convert frame to rgb_pixel matrix
        assign_image(img_matrix, cv_image<bgr_pixel>(frame));
        frames.push_back(img_matrix);
    }

    videoCapture.release();

    return frames;
}

matrix<rgb_pixel> HiCamera::captureFace(Hi *hi, int captureDeviceID, int maxFrames) {

    Mat frame;
    matrix<rgb_pixel> face;
    matrix<rgb_pixel> img_matrix;

    VideoCapture videoCapture(captureDeviceID); // open the default camera
    if (!videoCapture.isOpened()) throw runtime_error("Could not open camera.");

    // try as small as possible
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    for (size_t i = 0; i < maxFrames; i++) {
        videoCapture >> frame; // get a new frame from camera

        // convert frame to rgb_pixel matrix
        assign_image(img_matrix, cv_image<bgr_pixel>(frame));

        auto found_face = hi->findFace(img_matrix);
        if (found_face.size() != 0) { // only add frame if a face was found
            face = found_face;
            break;
        }
    }

    videoCapture.release();

    return face;
}
