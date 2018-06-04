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

    VideoCapture videoCapture(captureDeviceID); // open the default camera
    if (!videoCapture.isOpened()) throw runtime_error("Could not open camera.");

    std::vector<matrix<rgb_pixel>> frames;

    for (size_t i = 0; i < numFrames; i++) {
        Mat frame;
        videoCapture >> frame; // get a new frame from camera

        cv_image<bgr_pixel> img(frame);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, img);

        frames.push_back(matrix);
    }

    videoCapture.release();

    return frames;
}

matrix<rgb_pixel> HiCamera::captureFace(Hi &hi, int captureDeviceID, int maxFrames) {

    VideoCapture videoCapture(captureDeviceID); // open the default camera
    if (!videoCapture.isOpened()) throw runtime_error("Could not open camera.");

    // try as small as possible
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    Mat frame;
    matrix<rgb_pixel> face;
    matrix<rgb_pixel> img_matrix;

    for (size_t i = 0; i < maxFrames; i++) {
        if (i % 2 == 0) continue; // only capture every other frame

        videoCapture >> frame; // get a new frame from camera

        cv_image<bgr_pixel> img(frame);
        assign_image(img_matrix, img);

        auto found_face = hi.findFace(img_matrix);
        if (found_face.size() != 0) { // only add frame if a face was found
            face = found_face;
            break;
        }
    }

    videoCapture.release();

    return face;
}
