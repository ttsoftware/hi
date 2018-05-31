#include <iostream>
#include <experimental/filesystem>
#include <chrono>
#include <dlib/gui_widgets.h>

#include "Hi.h"
#include "HiCamera.h"

//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace std::chrono;

int main() try {

    // init time
    auto time0 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    // initialize neural networks
    auto hi = Hi();
    auto time1 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Loaded neural networks in " << time1 - time0 << " ms" << endl;

    auto frames = HiCamera::capture();

    cout << frames.size() << endl;

    image_window win;

    std::vector<pair<matrix<rgb_pixel>, rectangle>> framesFace;
    for (auto frame : frames) {

        std::map<rectangle, matrix<rgb_pixel>> face_map = hi.findFaces(frame);
        std::vector<rectangle> face_rectangles;
        face_rectangles.reserve(face_map.size());

        transform(begin(face_map), end(face_map), back_inserter(face_rectangles),
                  [](auto const &pair) {
                      return pair.first;
                  });

        framesFace.emplace_back(frame, face_rectangles[0]);
    }

    while (true) {
        for (auto frameFace : framesFace) {
            win.set_image(frameFace.first);
            win.clear_overlay();
            win.add_overlay(frameFace.second);

            usleep(static_cast<__useconds_t>(1e5));
        }
    }

/*auto descriptor = hi.createDescriptor("data/faces/troels.png", 100);
    cout << "Found a face descriptor." << endl;
    hi.storeDescriptor(descriptor, "data/face_descriptors/troels.dat");
    return 0;*//*

    auto troels_descriptor = hi.loadDescriptor("data/face_descriptors/troels.dat");
    auto time2 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Loaded Troels face descriptor in " << time2 - time1 << " ms" << endl;

    auto testfolder = std::experimental::filesystem::u8path("data/faces/test");
    for (auto &path : std::experimental::filesystem::directory_iterator(testfolder)) {

        if (path.path().has_extension()) { // don't compare directories

            auto time3 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

            cout << "\nComparing with " << path << endl;

            auto descriptors = hi.getDescriptors(path.path().string());
            auto result = hi.contains(descriptors, troels_descriptor, 0.5);

            if (result) {
                cout << "Troels is contained in " << path << endl;
            } else {
                cout << "Troels is not contained in " << path << endl;
            }

            auto time4 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

            cout << "Comparison took " << time4 - time3 << " ms." << endl;
        }
    }*/

    return 0;
}
catch (std::exception &e) {
    cout << e.what() << endl;
}