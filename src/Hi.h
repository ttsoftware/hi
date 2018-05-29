#ifndef HI_HI_H
#define HI_HI_H

#include <dlib/matrix.h>

using namespace dlib;
using namespace std;

class Hi {
public:
    Hi();

    matrix<float, 0, 1> createDescriptor(string imgLocation, int numJitters = 100);

    void storeDescriptor(matrix<float, 0, 1> face_descriptor, string filepath);

private:
    std::vector<matrix<rgb_pixel>> jitter(matrix<rgb_pixel> &img, int rounds);
};


#endif //HI_HI_H
