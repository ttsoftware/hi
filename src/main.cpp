#include <iostream>
#include <experimental/filesystem>
#include <chrono>

#include "Hi.h"

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

    /*auto descriptor = hi.createDescriptor("data/faces/troels.png", 25);

    cout << "Found a face descriptor." << endl;

    hi.storeDescriptor(descriptor, "data/face_descriptors/troels.dat");
    */

    auto troels_descriptor = hi.loadDescriptor("data/face_descriptors/troels.dat");

    auto time2 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << "Loaded Troels face descriptor in " << time2 - time1 << " ms" << endl;

    auto testfolder = std::experimental::filesystem::u8path("data/faces/test");
    for (auto &path : std::experimental::filesystem::directory_iterator(testfolder)) {

        if (path.path().has_extension()) { // don't compare directories

            auto time3 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

            cout << "\nComparing with " << path << endl;

            auto descriptors = hi.getDescriptors(path.path().string());

            // cout << "Found " << descriptors.size() << " other face descriptors." << endl;

            auto result = hi.contains(descriptors, troels_descriptor, 0.5);

            if (result) {
                cout << "Troels is contained in " << path << endl;
            } else {
                cout << "Troels is not contained in " << path << endl;
            }

            auto time4 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

            cout << "Comparison took " << time4 - time3 << " ms." << endl;
        }
    }

    return 0;
}
catch (std::exception &e) {
    cout << e.what() << endl;
}