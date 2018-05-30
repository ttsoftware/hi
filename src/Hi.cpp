#include "Hi.h"

#include <thread>
#include <dlib/dnn.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/serialize.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

// The following defines a ResNet network.
// The dlib_face_recognition_resnet_model_v1 model used here was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
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

frontal_face_detector detector;
shape_predictor sp;
anet_type net;

Hi::Hi() {
    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    detector = get_frontal_face_detector();

    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    deserialize("data/models/shape_predictor_68_face_landmarks.dat") >> sp;

    // And finally we load the DNN responsible for face recognition.
    deserialize("data/models/dlib_face_recognition_resnet_model_v1.dat") >> net;
}

std::vector<matrix<rgb_pixel>> Hi::jitter(matrix<rgb_pixel> &img, int rounds) {
    thread_local dlib::rand rnd;
    std::vector<matrix<rgb_pixel>> crops;

    for (int i = 0; i < rounds; ++i) {
        crops.push_back(jitter_image(img, rnd));
    }

    return crops;
}

std::vector<matrix<rgb_pixel>> Hi::findFaces(matrix<rgb_pixel> &img) {
    // Run the face detector on the image, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img)) {
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(sp(img, face), 150, 0.25), face_chip);
        faces.push_back(move(face_chip));
    }

    return faces;
}

matrix<float, 0, 1> Hi::createDescriptor(string img_location, int num_jitters) {
    matrix<rgb_pixel> img;
    load_image(img, img_location);
    std::vector<matrix<rgb_pixel>> faces = findFaces(img);

    if (faces.size() == 1) {
        // Use DNN to convert each face image in faces into a 128D std::vector.
        // In this 128D std::vector space, images from the same person will be close to each other
        // but std::vectors from different people will be far apart.  So we can use these std::vectors to
        // identify if a pair of images are from the same person or from different people.
        return mean(mat(net(jitter(faces[0], num_jitters))));
    }
    throw runtime_error("Invalid number of faces in image.");
}

void Hi::storeDescriptor(matrix<float, 0, 1> face_descriptor, string filepath) {
    ofstream input(filepath);
    serialize(face_descriptor, input);
}

matrix<float, 0, 1> Hi::loadDescriptor(string filepath) {
    matrix<float, 0, 1> face_descriptor;
    ifstream input(filepath);
    deserialize(face_descriptor, input);
    return face_descriptor;
}

bool Hi::contains(std::vector<matrix<float, 0, 1>> face_descriptors,
                  matrix<float, 0, 1> face_reference,
                  float confidence_threshold) {

    bool has_face = false;
    for (size_t i = 0; i < face_descriptors.size(); ++i) {

        std::cout << "Reference face is " << length(face_reference - face_descriptors[i]) << " from incoming face " << i << endl;

        if (length(face_reference - face_descriptors[i]) < confidence_threshold) {
            has_face = true;
            break;
        }
    }

    return has_face;
}

std::vector<matrix<float, 0, 1>> Hi::getDescriptors(string img_location) {
    matrix<rgb_pixel> img;
    load_image(img, img_location);
    // Convert each face image in faces into a 128D std::vector.
    return net(findFaces(img));
}
