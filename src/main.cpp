#include <iostream>
#include "Hi.h"

//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main() try {

    // initialize neural networks
    auto hi = Hi();
    auto descriptor = hi.createDescriptor("data/faces/troels.png", 25);

    cout << "Found a face descriptor." << endl;

    hi.storeDescriptor(descriptor, "data/face_descriptors/troels.dat");

    return 0;
}
catch (std::exception &e) {
    cout << e.what() << endl;
}