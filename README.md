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

In order to compile the code, you'll need to have installed the following libraries: `OpenCV`, `OpenCL`, `libusb` and `libfreenect`. I've prepared a script to do this for you.

```bash
git clone https://gist.github.com/113ae06addaa96444693.git
cd 113ae06addaa96444693
chmod +x install.bash
./install.bash
```

Compilation
===========

You can use the following to compile the code and run the exported application.

```bash
# cd into the project's folder
mkdir build
cd build
cmake ..
make
./kinectFilter
```

Attribution
===========

This code was based on the `cppview` example of libfreenect's C++ Wrapper API.
