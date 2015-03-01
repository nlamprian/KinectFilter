KinectFilter
============

This project brings together the following libraries: `libfreenect`, `OpenCV`, `OpenGL`, `OpenCL`.

`KinectFilter` offers a number of applications that use a Kinect sensor as a camera, process the data stream from Kinect on the GPU with OpenCL, and display the processed stream in a graphical window.

There are 3 examples in which a Laplacian of Gaussian (LoG) filter is applied on the RGB stream, and another one in which a 3D point cloud is built while performing RGB normalizaton. The applications are interactive, so you can examine the effects of the filters on the incoming streams. You can find a demonstration of how the application from version 1.0 is performing on [YouTube](https://www.youtube.com/watch?v=jnuAnIt9vFY).

![snapshot](http://i76.photobucket.com/albums/j16/paign10/snapshot_zps0b51dlju.jpg)

The Laplacian filter is an edge detection operator that works on both the `x` and `y` image axes. The Gaussian filter is a simple smoothing operator which unfortunately smooths also the edges. In the demo applications, you will be able to examine the effects of the Gaussian filter on the edge detection process in real-time and get a better understanding of the matter. For a more superior, edge-preserving, smoothing operator, take a look at the [Bilateral](http://en.wikipedia.org/wiki/Bilateral_filter) and the [Guided Image](http://research.microsoft.com/en-us/um/people/kahe/eccv10/) filters.

Note
====

The code was tested on Ubuntu 12.04/14.04, on a system with an AMD GPU.

**New on version 2.0:**

* Cleaned up code
* Included example on the OpenCL C++ API
* Included 2 examples on the OpenCL-OpenGL interoperability (based on the OpenCL C++ API)

Dependencies
============

In order to compile the code, you'll need to have installed the following libraries: `OpenGL`, `GLUT`, `GLEW`, `OpenCV`, `OpenCL`, `libusb` and `libfreenect`. I've prepared a script to do this for you.

```bash
git clone https://gist.github.com/113ae06addaa96444693.git
bash 113ae06addaa96444693/kinect_filter_dependencies.bash
rm -r 113ae06addaa96444693
```

Compilation
===========

You can use the following to compile the code and run the exported applications.

```bash
git clone https://github.com/pAIgn10/KinectFilter.git
cd KinectFilter
mkdir build
cd build
cmake ..
make
./bin/kinectFilter_clc
./bin/kinectFilter_clc++
./bin/kinectFilter_gl_interop_texture
./bin/kinectFilter_gl_interop_vertex_buffer
```

Attribution
===========

Version 1.0 of this project was based on the `cppview` example of libfreenect's C++ Wrapper API.
