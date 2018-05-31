#include <iostream>
#include <experimental/filesystem>
#include <chrono>
#include <dlib/gui_widgets.h>

#include "Hi.h"
#include "HiCamera.h"

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

    cout << "Loaded Troels face descriptor in " << time2 - time1 << " ms" << endl;

    auto time3 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    auto frames = HiCamera::capture(0, 5);

    // find first face in all frames
    std::vector<matrix<rgb_pixel>> faces;
    for (auto frame : frames) {
        auto found_faces = hi.findFaces(frame);
        if (!found_faces.empty()) { // only add frame if a face was found
            faces.emplace_back(found_faces[0]);
            break;
        }
    }

    if (faces.empty()) return 1;

    auto time4 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Capturing and finding " << faces.size() << " faces in " << time4 - time3 << " ms." << endl;

    auto descriptors = hi.getDescriptors(faces[0]);

    auto time5 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Creating faces descriptors in " << time5 - time4 << " ms." << endl;

    auto result = hi.contains(descriptors, troels_descriptor, 0.5);

    auto time6 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    if (result) {
        cout << "Troels is contained in frames" << endl;
    } else {
        cout << "Troels is not contained in frames" << endl;
    }

    cout << "Comparison took " << time6 - time5 << " ms." << endl;

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
    auto hi = Hi();

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
    }
}