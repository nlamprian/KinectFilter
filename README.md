KinectFilter
============

This is an example application that brings together the following libraries: `libfreenect`, `OpenCV`, `OpenCL`, `OpenGL`.

It uses a Kinect sensor as a camera. It gets the video stream from Kinect and then applies a Laplacian filter on it. This process happens on the GPU with OpenCL. The processed stream is displayed on the screen. You can find a demonstration of how the application is performing on [YouTube](https://www.youtube.com/watch?v=jnuAnIt9vFY).

![snapshot](http://i859.photobucket.com/albums/ab154/lampnick67/kinectfilter_zps6695c598.png)

Note
====

The code was tested on Ubuntu 12.04, on a system with an AMD GPU.

Installs
========

In order to compile the code, you'll need to have installed the following libraries.

* [OpenCV](https://help.ubuntu.com/community/OpenCV)
* [OpenCL](http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-tools-sdks/amd-accelerated-parallel-processing-app-sdk/) (If you have another vendor's GPU, look for their own OpenCL implementation)
* [libfreenect](https://github.com/OpenKinect/libfreenect)


Modifications
=============

I used the distribution packages to install `libfreenect`. There was an issue with the `libfreenect.hpp` header file, and I ended up replacing it with the one from the git repository.

```bash 
cd
git clone https://github.com/OpenKinect/libfreenect.git
sudo mv /usr/include/libfreenect.hpp /usr/include/libfreenect.hpp.backup
sudo cp ~/libfreenect/wrappers/cpp/libfreenect.hpp /usr/include/
```

Compilation
===========

You can use the following to compile the code and run the exported application.

```bash
g++ -std=c++0x -I/usr/local/include/libusb-1.0 -I/opt/AMDAPP/include `pkg-config --cflags opencv` kinectFilter.cpp -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lGL -lglut -lfreenect `pkg-config --libs opencv` -o kinectFilter
./kinectFilter &> /dev/null
```

Attribution
===========

This code was based on the `cppview` example of libfreenect's C++ Wrapper API.
