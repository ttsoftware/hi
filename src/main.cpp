#include <iostream>
#include <filesystem>
#include <chrono>

#include "Hi.h"

//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace std::chrono;
namespace fs = filesystem;

int main() try {

    // init time
    auto time0 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    // initialize neural networks
    auto hi = Hi();

    auto time1 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << time1 - time0 << ": loaded neural networks." << endl;

    /*auto descriptor = hi.createDescriptor("data/faces/troels.png", 25);

    cout << "Found a face descriptor." << endl;

    hi.storeDescriptor(descriptor, "data/face_descriptors/troels.dat");
    */

    auto troels_descriptor = hi.loadDescriptor("data/face_descriptors/troels.dat");

    auto time2 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

    cout << time2 - time1 << ": loaded Troels face descriptor: " << trans(troels_descriptor) << endl;

    auto testfolder = fs::u8path("data/faces/test");
    for (auto &path : fs::directory_iterator(testfolder)) {

        auto time3 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

        cout << "\nComparing with " << path << endl;

        auto descriptors = hi.getDescriptors(path.path().string());

        cout << "Found " << descriptors.size() << " other face descriptors." << endl;

        auto result = hi.contains(descriptors, troels_descriptor);

        if (result) {
            cout << "Troels is contained in " << path << endl;
        } else {
            cout << "Troels is not contained in " << path << endl;
        }

        auto time4 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

        cout << "Comparison took " << time4 - time3 << " ms." << endl;
    }

    return 0;
}
catch (std::exception &e) {
    cout << e.what() << endl;
}