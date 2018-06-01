#include "Hi.h"

#include <thread>
#include <typeinfo>
#include <dlib/dnn.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/serialize.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

Hi::Hi() {
    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    detector = get_frontal_face_detector();

    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    deserialize("data/models/shape_predictor_68_face_landmarks.dat") >> shape_corrector;

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

matrix<rgb_pixel> Hi::findFace(matrix<rgb_pixel> &img) {
    // Run the face detector on the image, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    matrix<rgb_pixel> face_chip;

    auto faces = detector(img);
    if (!faces.empty()) {
        rectangle face = faces[0];
        extract_image_chip(img, get_face_chip_details(shape_corrector(img, face), 150, 0.2), face_chip);
    }

    return face_chip;
}

std::vector<matrix<rgb_pixel>> Hi::findFaces(matrix<rgb_pixel> &img) {
    // Run the face detector on the image, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    std::vector<matrix<rgb_pixel>> faces;
    for (rectangle face : detector(img)) {
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape_corrector(img, face), 150, 0.2), face_chip);
        faces.push_back(move(face_chip));
    }

    return faces;
}

std::vector<rectangle> Hi::findFaceLocations(matrix<rgb_pixel> &img) {
    return detector(img);
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

    bool has_reference_face = false;
    for (const auto &face_descriptor : face_descriptors) {
        if (length(face_reference - face_descriptor) < confidence_threshold) {
            has_reference_face = true;
            break;
        }
    }

    return has_reference_face;
}

std::vector<matrix<float, 0, 1>> Hi::getDescriptors(matrix<rgb_pixel> &face) {
    // Convert each face image in faces into a 128D std::vector.
    const std::vector<matrix<rgb_pixel>> faces(1, face);
    return net(faces);
}
