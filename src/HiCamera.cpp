#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <zconf.h>

#include <dlib/dnn.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "HiCamera.h"

using namespace cv;
using namespace dlib;
using namespace std;

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template<int N, template<typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template<int N, typename SUBNET> using ares      = relu<residual<block, N, affine, SUBNET>>;
template<int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template<typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template<typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template<typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template<typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
                                                        input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;

void HiCamera::capture() {
    VideoCapture videoCapture(0); // open the default camera
    if (!videoCapture.isOpened()) throw runtime_error("Could not open camera.");

    std::vector<matrix<rgb_pixel>> frames;

    dlib::image_window win;

    for (size_t i = 0; i < 30; i++) {
        Mat frame;
        videoCapture >> frame; // get a new frame from camera

        cv_image<bgr_pixel> img(frame);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, img);

        frames.push_back(matrix);
    }

    videoCapture.release();

    anet_type net;
    shape_predictor sp;

    frontal_face_detector detector = get_frontal_face_detector();
    deserialize("data/models/shape_predictor_68_face_landmarks.dat") >> sp;
    deserialize("data/models/dlib_face_recognition_resnet_model_v1.dat") >> net;

    for (const auto &frame : frames) {
        win.set_image(frame);
        win.clear_overlay();

        for (auto face : detector(frame)) win.add_overlay(face);
    }
}