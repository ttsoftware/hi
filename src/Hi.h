#ifndef HI_HI_H
#define HI_HI_H

#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/serialize.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

class Hi {
public:
    Hi();

    /**
     * Returns a list of faces and their corresponding matrices found in image
     *
     * @param img
     * @return std::vector<matrix<rgb_pixel>>
     */
    std::vector<matrix<rgb_pixel>> findFaces(matrix<rgb_pixel> &img);

    /**
     * Returns the first face and its corresponding matrices found in image
     *
     * @param img
     * @return std::vector<matrix<rgb_pixel>>
     */
    matrix<rgb_pixel> findFace(matrix<rgb_pixel> &img);

    /**
     * For each face extract a rectangle describing its location
     *
     * @param img
     * @return std::vector<rectangle>
     */
    std::vector<rectangle> findFaceLocations(matrix<rgb_pixel> &img);

    /**
     * Finds all (simple) face descriptors in image.
     *
     * @param face image to create descriptors of
     * @return std::vector<matrix<float, 0, 1>>
     */
    std::vector<matrix<float, 0, 1>> getDescriptors(matrix<rgb_pixel> &face);

    /**
     * Creates a new detailed face descriptor from image. Assumes only one face in image.
     *
     * @param img_location location of image to create descriptor of
     * @return matrix<float, 0, 1>
     */
    matrix<float, 0, 1> createDescriptor(string img_location, int num_jitters = 100);

    /**
     * Stores a face descriptor matrix in a serialized form
     *
     * @param face_descriptor face descriptor matrix
     * @param filepath location of new file
     */
    void storeDescriptor(matrix<float, 0, 1> face_descriptor, string filepath);

    /**
     * Loads a store face descriptor matrix into memory
     *
     * @param filepath
     * @return matrix<float, 0, 1>
     */
    matrix<float, 0, 1> loadDescriptor(string filepath);

    /**
     * Returns true if face_descriptors approximately contains face_reference
     *
     * @param face_reference
     * @param face_descriptors
     * @param confidence_threshold lower values are better
     * @return bool
     */
    bool contains(std::vector<matrix<float, 0, 1>> face_descriptors,
                  matrix<float, 0, 1> face_reference,
                  float confidence_threshold = 0.5);

private:
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
    shape_predictor shape_corrector;
    anet_type net;

    /**
     * All this function does is make 100 copies of img, all slightly jittered by being
     * zoomed, rotated, and translated a little bit differently. They are also randomly
     * mirrored left to right.
     *
     * @param img image to jitter
     * @param rounds number of times to jitter
     * @return std::vector<matrix<rgb_pixel>>
     */
    std::vector<matrix<rgb_pixel>> jitter(matrix<rgb_pixel> &img, int rounds);
};


#endif //HI_HI_H
