#include <iostream>
#include <chrono>
#include <iterator>
#include <vector>
#include <thread>
#include <string>
#include <fstream>
#include <experimental/filesystem>

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

int recognize(const matrix<float, 0, 1> &reference_descriptor) {

    auto face = HiCamera::captureFace(&hi, 0, 10);
    if (face.size() == 0) return 11;

    auto descriptors = hi.getDescriptors(face);
    bool result = hi.contains(descriptors, reference_descriptor, 0.6);

    return result ? 0 : 1;
}

int main(int argc, char **argv) try {
    cout << "Loaded neural networks in " << time1 - time0 << " ms" << endl;

    if (argc == 1) {
        // start daemon
        pid_t parent_pid = getpid();
        pid_t child_pid = fork();

        if (child_pid == 0) {
            // load reference descriptors
            std::vector<matrix<float, 0, 1>> descriptors;
            auto descriptor_folder = std::experimental::filesystem::u8path("data/face_descriptors");
            for (auto &path : std::experimental::filesystem::directory_iterator(descriptor_folder)) {
                descriptors.push_back(hi.loadDescriptor(path.path().string()));
            }

            // open pipe and wait forever
            const char *fifo = "/tmp/hi_fifo";
            mkfifo(fifo, 0666);
            ifstream fifo_stream(fifo);
            fifo_stream.ignore();

            char* buffer[16];

            string command;
            while (true) {
                fifo_stream >> command;
                if (command.compare("auth")) {
                    // authenticate against existing stored descriptors
                    for (auto &descriptor : descriptors) {
                        if (recognize(descriptor) == 0) break;
                    }
                }

                // clear fifo for next command
                auto fd = fopen(fifo, "r");
                fread(buffer, 1, 15, fd);
                fclose(fd);
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
        }
    }

    return 1;
} catch (std::exception &e) {
    cout << e.what() << endl;
}
