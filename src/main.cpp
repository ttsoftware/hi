#include <iostream>
#include <experimental/filesystem>
#include <chrono>
#include <iterator>
#include <vector>
#include <thread>

#include <dlib/opencv.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/generic_image.h>
#include <dlib/image_transforms/equalize_histogram.h>
#include <dlib/gui_widgets.h>

#include "Hi.h"
#include "HiCamera.h"

using namespace dlib;
using namespace std;
using namespace std::this_thread;
using namespace std::chrono;

// init time
auto time0 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

// initialize neural networks
auto hi = Hi();
auto time1 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

int recognize(const string &reference_descriptor_name) try {

    cout << "Loaded neural networks in " << time1 - time0 << " ms" << endl;

    auto troels_descriptor = hi.loadDescriptor("data/face_descriptors/" + reference_descriptor_name + ".dat");
    auto time2 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Loaded reference face descriptor in " << time2 - time1 << " ms" << endl;

    auto face = HiCamera::captureFace(hi, 0, 10);

    auto time3 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    if (face.size() == 0) return 11;

    cout << "Capturing and finding a face in " << time3 - time2 << " ms." << endl;

    auto time4 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    auto descriptors = hi.getDescriptors(face);
    bool result = hi.contains(descriptors, troels_descriptor, 0.5);

    auto time5 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();
    cout << "Creating face descriptors in " << time5 - time4 << " ms." << endl;
    cout << (result ? "Success" : "Failure") << endl;
    cout << "Total time " << (time5 - time0) / 1000 << " s." << endl;

    return 0;
}
catch (std::exception &e) {
    cout << e.what() << endl;
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        // start daemon
        return 1;
    }
    if (argc == 2) {
        // recognize face
        return recognize(string(argv[1]));
    }
    if (argc == 3) {
        // add a face
        if (string(argv[1]).compare("add") == 0) {

            cout << "Capturing face please sit still and stare directly into the camera." << endl;
            sleep_until(system_clock::now() + seconds(1));
            cout << "3" << endl;
            sleep_until(system_clock::now() + seconds(1));
            cout << "2" << endl;
            sleep_until(system_clock::now() + seconds(1));
            cout << "1" << endl;
            sleep_until(system_clock::now() + seconds(1));

            auto frames = HiCamera::capture(0, 15);

            cout << "Captured face..." << endl;

            bool has_face = false;
            for (auto frame : frames) {
                auto face = hi.findFace(frame);
                if (face.size() > 0) {
                    cout << "Creating unique face descriptor vector..." << endl;
                    hi.storeDescriptor(hi.createDescriptor(face), "data/face_descriptors/" + string(argv[2]) + ".dat");
                    has_face = true;
                    break;
                }
            }
            if (!has_face) {
                cout << "Could not find a face in the captured frames." << endl;
                return 1;
            }
            cout << "Done!" << endl;
        } else {
        }
    }

    return 1;
}