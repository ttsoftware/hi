#include <iostream>
#include <experimental/filesystem>
#include <chrono>

#include <dlib/opencv.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/generic_image.h>
#include <dlib/image_transforms/equalize_histogram.h>
#include <dlib/gui_widgets.h>

#include "Hi.h"
#include "HiCamera.h"

using namespace dlib;
using namespace std;
using namespace std::chrono;

int main() try {

    // init time
    auto time0 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    // initialize neural networks
    auto hi = Hi();
    auto time1 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Loaded neural networks in " << time1 - time0 << " ms" << endl;

    auto troels_descriptor = hi.loadDescriptor("data/face_descriptors/troels.dat");
    auto time2 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Loaded reference face descriptor in " << time2 - time1 << " ms" << endl;

    auto face = HiCamera::captureFace(hi, 0, 10);

    auto time3 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Capturing and finding a face in " << time3 - time2 << " ms." << endl;

    if (face.size() == 0) return 11;

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
}

void test() {

    /*
    std::vector<pair<matrix<rgb_pixel>, rectangle>> framesFace;
     for (auto frame : frames) {
        auto face_rectangles = hi.findFaceLocations(frame);
        if (!face_rectangles.empty()) { // only add frame if a face was found
            framesFace.emplace_back(frame, face_rectangles[0]);
        }
    }
    image_window win;
    while (true) {
        for (auto frameFace : framesFace) {
            win.set_image(frameFace.first);
            win.clear_overlay();
            win.add_overlay(frameFace.second);

            usleep(static_cast<__useconds_t>(1e5));
        }
    }*/

    /*
    auto descriptor = hi.createDescriptor("data/faces/troels.png", 100);
    cout << "Found a face descriptor." << endl;
    hi.storeDescriptor(descriptor, "data/face_descriptors/troels.dat");
    return 0;
    */

    // initialize neural networks
    /*auto hi = Hi();

    auto troels_descriptor = hi.loadDescriptor("data/face_descriptors/troels.dat");

    auto testfolder = std::experimental::filesystem::u8path("data/faces/test");
    for (auto &path : std::experimental::filesystem::directory_iterator(testfolder)) {

        if (path.path().has_extension()) { // don't compare directories

            auto time0 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

            cout << "\nComparing with " << path << endl;

            auto descriptors = hi.getDescriptors(path.path().string());
            auto result = hi.contains(descriptors, troels_descriptor, 0.5);

            if (result) {
                cout << "Troels is contained in " << path << endl;
            } else {
                cout << "Troels is not contained in " << path << endl;
            }

            auto time1 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

            cout << "Comparison took " << time1 - time0 << " ms." << endl;
        }
    }*/
}