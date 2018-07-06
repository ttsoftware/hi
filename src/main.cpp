#include <iostream>
#include <chrono>
#include <iterator>
#include <vector>
#include <thread>
#include <string>
#include <fstream>
#include <experimental/filesystem>

#include <sys/stat.h>

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

auto CAPTURE_DEVICE = 1;
// config dir
// auto hi_path = "/lib/security/hi";
auto hi_path = ((string)getenv("HOME")) + "/.hi";

// init time
auto time0 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

// initialize neural networks
auto hi = Hi(hi_path);
auto time1 = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();

int recognize(const matrix<float, 0, 1> &reference_descriptor) {

    auto face = HiCamera::captureFace(&hi, CAPTURE_DEVICE, 3);
    if (face.size() == 0) return 10;

    auto descriptors = hi.getDescriptors(face);
    bool result = hi.contains(descriptors, reference_descriptor, 0.35);

    return result ? 0 : 11;
}

int main(int argc, char **argv) try {
    cout << "Loaded neural networks in " << time1 - time0 << " ms" << endl;

    if (argc == 1) {
        // start daemon
        pid_t parent_pid = getpid();
        pid_t child_pid = fork();

        if (child_pid == 0) {
            // create config folders if not exist
            struct stat sb;
            if (stat(hi_path.c_str(), &sb) == -1) {
                system(("mkdir -p " + hi_path + "/models").c_str());
                system(("mkdir -p " + hi_path + "/face-descriptors").c_str());
            }

            // load reference descriptors
            std::vector<matrix<float, 0, 1>> descriptors;
            auto descriptor_folder = std::experimental::filesystem::u8path(hi_path + "/face-descriptors");
            for (auto &path : std::experimental::filesystem::directory_iterator(descriptor_folder)) {
                descriptors.push_back(hi.loadDescriptor(path.path().string()));
            }

            // open pipe and wait forever
            const char *fifo_in = "/tmp/hi_fifo_in";
            const char *fifo_out = "/tmp/hi_fifo_out";
            mkfifo(fifo_in, 0666);
            mkfifo(fifo_out, 0666);

            string command;
            while (true) {
                ofstream fifo_ostream(fifo_out, std::ofstream::out);
                ifstream fifo_istream(fifo_in, std::ofstream::in);
                fifo_istream.ignore();

                fifo_istream >> command;
                if (command.compare("auth")) {
                    // authenticate against existing stored descriptors
                    for (auto &descriptor : descriptors) {
                        fifo_ostream << recognize(descriptor);
                    }
                }

                fifo_ostream.close();
            }
        } else {
            cout << "Successfully started Hi daemon on pid " << child_pid << endl;
        }

        return 0;
    }
    if (argc == 2) {
        // recognize face
        return recognize(hi.loadDescriptor(string(argv[1])));
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

            auto frames = HiCamera::capture(CAPTURE_DEVICE, 15);

            cout << "Captured face..." << endl;

            bool has_face = false;
            for (auto frame : frames) {
                auto face = hi.findFace(frame);
                if (face.size() > 0) {
                    cout << "Creating unique face descriptor vector..." << endl;
                    hi.storeDescriptor(hi.createDescriptor(face), hi_path + "/face-descriptors/" + string(argv[2]) + ".dat");
                    has_face = true;
                    break;
                }
            }
            if (!has_face) {
                cout << "Could not find a face in the captured frames." << endl;
                return 1;
            }
            cout << "Done!" << endl;
        }
    }

    return 1;
} catch (std::exception &e) {
    cout << e.what() << endl;
}
