#ifndef HI_HI_H
#define HI_HI_H

#include <dlib/matrix.h>

using namespace dlib;
using namespace std;

class Hi {
public:
    Hi();

    /**
     * Finds all (simple) face descriptors in image.
     *
     * @param img_location location of image to create descriptors
     * @return std::vector<matrix<float, 0, 1>>
     */
    std::vector<matrix<float, 0, 1>> getDescriptors(string img_location);

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

    /**
     * Returns a list of faces matrices found in image matrix
     *
     * @param img
     * @return std::vector<matrix<rgb_pixel>>
     */
    std::vector<matrix<rgb_pixel>> findFaces(matrix<rgb_pixel> &img);
};


#endif //HI_HI_H
