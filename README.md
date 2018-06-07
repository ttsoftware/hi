# hi
Facial recognition authentication for linux PAM.
This is a port of Howdy (https://github.com/Boltgolt/howdy) to c++ and native dlib interaction in order to speed up performance.
The main benefit is that Hi runs as a daemon, thus reducing memory overhead and removing the need to constantly load the neural networks into memory.
Hi cuts the authentication time in half while increasing accuracy.
