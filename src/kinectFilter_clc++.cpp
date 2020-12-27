/**
 * Name: KinectFilter
 * Author: Nick Lamprianidis <nlamprian@gmail.com>
 * Version: 2.0
 * Description: This project brings together the following libraries: 
 *              libfreenect, OpenCV, OpenGL, OpenCL. There is a number of 
 *              applications that use a Kinect sensor as a camera, process the 
 *              data stream from Kinect on the GPU with OpenCL, and display the 
 *              processed stream in a graphical window.
 * Source: https://github.com/nlamprian/KinectFilter
 * License: Copyright (c) 2014-2015 Nick Lamprianidis
 *          This code is licensed under the GPL v2 license
 * Attribution: This application was based on the "cppview" example of the
 *              libfreenect's C++ wrapper API.
 *              That example is part of the OpenKinect Project. www.openkinect.org
 *              Copyright (c) 2010 individual OpenKinect contributors
 *              See the CONTRIB file for details.
 *
 * Filename: kinectFilter_clc++.cpp
 * File description: Example application utilizing the OpenCL C++ API.
 *                   It applies a LoG filter on the Kinect video stream, and 
 *                   allows to examine, online, the results of the filtering.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <GL/glew.h>

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#include <GLUT/glut.h>
#else
#include <CL/cl.hpp>
#include <GL/glut.h>
#endif
#include <libfreenect.hpp>


// Window parameters
const int gl_win_width = 640;
const int gl_win_height = 480;
int glWinId;

// GL texture ID
GLuint glRGBTex;

// Freenect
class MyFreenectDevice;
Freenect::Freenect freenect;
MyFreenectDevice *device;
double freenectAngle = 0;

// OpenCL
class Filter;
Filter *opencl;


// A class for filtering an image on the GPU
class Filter
{
public:
    Filter () : global { gl_win_width, gl_win_height }, smoothed (true)
    {
        // Image region for transfers
        region[0] = gl_win_width;
        region[1] = gl_win_height;
        region[2] = 1;

        // Image dimensions
        const int width = gl_win_width;
        const int height = gl_win_height;

        //! Applying multiple times a box filter, approximates a Gaussian filter
        const float box_filter[] = { 0.125f, 0.125f, 0.125f,
                                     0.125f, 0.125f, 0.125f,
                                     0.125f, 0.125f, 0.125f };
        const float laplacian_filter[] = { 1.f,  1.f, 1.f,
                                           1.f, -8.f, 1.f,
                                           1.f,  1.f, 1.f };
        const int filterWidth = 3;
        const int filterSize = filterWidth * filterWidth * sizeof (float);

        // Get the list of platforms
        cl::Platform::get (&platforms);

        // Get the GPU devices in the first platform
        platforms[0].getDevices (CL_DEVICE_TYPE_GPU, &devices);

        // Create a context for the first device
        context = cl::Context (devices[0]);

        // Create a command queue for the device
        queue = cl::CommandQueue (context, devices[0]);

        // Create an image sampler
        sampler = cl::Sampler (context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);

        // Create an image format
        cl::ImageFormat format (CL_R, CL_UNSIGNED_INT8);

        // Create an image instance for the source image on the device
        bufferSourceImage = cl::Image2D (context, CL_MEM_READ_ONLY, format, width, height);

        // Create image instances for the intermediate results on the device
        bufferInterImage1 = cl::Image2D (context, CL_MEM_READ_WRITE, format, width, height);
        bufferInterImage2 = cl::Image2D (context, CL_MEM_READ_WRITE, format, width, height);

        // Create an image instance for the output image on the device
        bufferOutputImage = cl::Image2D (context, CL_MEM_WRITE_ONLY, format, width, height);

        // Create buffers for the filters on the device
        bufferBoxFilter = cl::Buffer (context, CL_MEM_READ_ONLY, filterSize);
        bufferLaplacianFilter = cl::Buffer (context, CL_MEM_READ_ONLY, filterSize);

        // Copy the filters to the device
        queue.enqueueWriteBuffer (bufferBoxFilter, CL_FALSE, 0, filterSize, box_filter);
        queue.enqueueWriteBuffer (bufferLaplacianFilter, CL_FALSE, 0, filterSize, laplacian_filter);

        // Read the program source
        std::ifstream sourceFile ("kernels/kernels.cl");
        std::string programCode (std::istreambuf_iterator<char> (sourceFile), (std::istreambuf_iterator<char> ()));
    
        // Create a program
        cl::Program::Sources source (1, std::make_pair (programCode.c_str (), programCode.length () + 1));
        program = cl::Program (context, source);

        try
        {
            // Compile the program
            program.build (devices);
        }
        catch (const cl::Error &error)
        {
            std::cerr << error.what () << " ("
                      << error.err ()  << ")"  << std::endl;
            
            std::string log;
            program.getBuildInfo (devices[0], CL_PROGRAM_BUILD_LOG, &log);
            std::cout << log << std::endl;

            exit (EXIT_FAILURE);
        }

        // Create kernel
        kernelConv = cl::Kernel (program, "convolution");

        // Set common kernel arguments
        kernelConv.setArg (2, height);
        kernelConv.setArg (3, width);
        kernelConv.setArg (5, filterWidth);
        kernelConv.setArg (6, sampler);
    }

    void convolve (std::vector<uint8_t> &image)
    {
        // Copy the source image to the device
        queue.enqueueWriteImage (bufferSourceImage, CL_FALSE, origin, region, 0, 0, image.data ());

        if (smoothed)
        {
            kernelConv.setArg (0, bufferSourceImage);
            kernelConv.setArg (1, bufferInterImage1);
            kernelConv.setArg (4, bufferBoxFilter);

            // Apply the first box firter
            queue.enqueueNDRangeKernel (kernelConv, cl::NullRange, global, cl::NullRange);

            kernelConv.setArg (0, bufferInterImage1);
            kernelConv.setArg (1, bufferInterImage2);

            // Apply the second box firter
            queue.enqueueNDRangeKernel (kernelConv, cl::NullRange, global, cl::NullRange);

            kernelConv.setArg (0, bufferInterImage2);
        }
        else
        {
            kernelConv.setArg (0, bufferSourceImage);
        }

        kernelConv.setArg (1, bufferOutputImage);
        kernelConv.setArg (4, bufferLaplacianFilter);

        // Perform the edge detection
        queue.enqueueNDRangeKernel (kernelConv, cl::NullRange, global, cl::NullRange);

        // Read back the output image
        queue.enqueueReadImage (bufferOutputImage, CL_TRUE, origin, region, 0, 0, image.data ());
    }

    // Returns the state of the flag for smoothing
    bool smoothing ()
    {
        return smoothed;
    }

    // Toggles the flag for applying a Gaussian fitler on the frames
    // Returns the new state of the flag
    bool toggleSmoothing ()
    {   
        smoothed = !smoothed;
        return smoothed;
    }

private:
    // Image transfer parameters
    cl::size_t<3> origin;
    cl::size_t<3> region;

    // Workspace dimensions
    cl::NDRange global;

    bool smoothed;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Sampler sampler;
    cl::Image2D bufferSourceImage, bufferOutputImage;
    cl::Image2D bufferInterImage1, bufferInterImage2;
    cl::Buffer bufferBoxFilter, bufferLaplacianFilter;
    cl::Program program;
    cl::Kernel kernelConv;
};


// A class hierarchy for manipulating a mutex
class Mutex
{
public:
    void lock () { freenectMutex.lock (); }
    void unlock () { freenectMutex.unlock (); }

    // A class that automates the manipulation of 
    // the outer class instance's mutex.
    // Mutex's mutex is locked with the creation of a ScopedLock instance 
    // and unlocked with the destruction of the ScopedLock instance.
    class ScopedLock
    {
    public:
        ScopedLock (Mutex &mtx) : mMutex (mtx) { mMutex.lock (); }
        ~ScopedLock () { mMutex.unlock (); }

    private:
        Mutex &mMutex;
    };

private:
    std::mutex freenectMutex;
};


// A class that extends Freenect::FreenectDevice by defining the VideoCallback
// callback function so we can get updates with the latest RGB frame
class MyFreenectDevice : public Freenect::FreenectDevice
{
public:
    MyFreenectDevice (freenect_context *ctx, int index)
        : Freenect::FreenectDevice (ctx, index),
          rgbBuffer (freenect_find_video_mode (FREENECT_RESOLUTION_MEDIUM, 
                                                    FREENECT_VIDEO_RGB).bytes),
          newRGBFrame (false)
    {
    }

    // Delivers the latest RGB frame
    // Do not call directly, it's only used by the library
    void VideoCallback (void *rgb, uint32_t timestamp)
    {
        Mutex::ScopedLock lock (rgbMutex);

        std::copy (static_cast<uint8_t *> (rgb), 
                   static_cast<uint8_t *> (rgb) + getVideoBufferSize (), 
                   rgbBuffer.begin ());
        newRGBFrame = true;
    };

    // Delivers the most recently received frame after filtering it
    bool getRGB (std::vector<uint8_t> &buffer)
    {
        Mutex::ScopedLock lock (rgbMutex);

        if (!newRGBFrame)
            return false;

        // Transform the frame to gray-scale
        // OpenCV was used just for experimenting
        // The operation could just as easily be done on the GPU
        cv::Mat gray, rgb (gl_win_height, gl_win_width, CV_8UC3, rgbBuffer.data ());
        cv::cvtColor (rgb, gray, CV_RGB2GRAY);
        std::copy (gray.datastart, gray.dataend + 1, buffer.data ());

        // Apply the filters to the image
        opencl->convolve (buffer);

        newRGBFrame = false;

        return true;
    }

private:
    Mutex rgbMutex;
    std::vector<uint8_t> rgbBuffer;
    bool newRGBFrame;
};


// Display callback for the window
void drawGLScene ()
{
    static std::vector<uint8_t> image (gl_win_width * gl_win_height);

    device->getRGB (image);

    glClear (GL_COLOR_BUFFER_BIT);

    glBegin (GL_QUADS);
    glColor4f (1.f, 1.f, 1.f, 1.f);
    glVertex2i (0, 0); glTexCoord2f (1.f, 0.f);
    glVertex2i (gl_win_width, 0); glTexCoord2f (1.f, 1.f);
    glVertex2i (gl_win_width, gl_win_height); glTexCoord2f (0.f, 1.f);
    glVertex2i (0, gl_win_height); glTexCoord2f (0.f, 0.f);
    glEnd ();

    glEnable (GL_TEXTURE_2D);
    // glBindTexture (GL_TEXTURE_2D, glRGBTex);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_LUMINANCE, gl_win_width, gl_win_height, 0,
                  GL_LUMINANCE, GL_UNSIGNED_BYTE, image.data ());

    std::ostringstream state;
    state << "Smoothing: " << (opencl->smoothing () ? "ON" : "OFF");

    glRasterPos2i (530, 30);
    for (auto c : state.str ())
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);

    glutSwapBuffers ();
}


// Idle callback for the window
void idleGLScene ()
{
    glutPostRedisplay ();
}


// Reshape callback for the window
void resizeGLScene (int width, int height)
{
    glViewport (0, 0, width, height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glOrtho (0.0, gl_win_width, gl_win_height, 0.0, -1.0, 1.0);
    glMatrixMode (GL_MODELVIEW);
}


// Keyboard callback for the window
void keyPressed (unsigned char key, int x, int y)
{
    switch (key)
    {
        case 0x1B:  // ESC
        case  'Q':
        case  'q':
            glutDestroyWindow (glWinId);
            break;
        case 'F':
        case 'f':
            opencl->toggleSmoothing ();
            break;
        case  'W':
        case  'w':
            if (++freenectAngle > 30)
                freenectAngle = 30;
            device->setTiltDegrees (freenectAngle);
            break;
        case  'S':
        case  's':
            if (--freenectAngle < -30)
                freenectAngle = -30;
            device->setTiltDegrees (freenectAngle);
            break;
        case  'R':
        case  'r':
            freenectAngle = 0;
            device->setTiltDegrees (freenectAngle);
            break;
        case  '1':
            device->setLed (LED_GREEN);
            break;
        case  '2':
            device->setLed (LED_RED);
            break;
        case  '3':
            device->setLed (LED_YELLOW);
            break;
        case  '4':
        case  '5':
            device->setLed (LED_BLINK_GREEN);
            break;
        case  '6':
            device->setLed (LED_BLINK_RED_YELLOW);
            break;
        case  '0':
            device->setLed (LED_OFF);
            break;
    }
}


// Initializes GLUT
void initGL (int argc, char **argv)
{
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize (gl_win_width, gl_win_height);
    glutInitWindowPosition ((glutGet (GLUT_SCREEN_WIDTH) - gl_win_width) / 2,
                            (glutGet (GLUT_SCREEN_HEIGHT) - gl_win_height) / 2 - 70);
    glWinId = glutCreateWindow ("KinectFilter - OpenCL C++ API");

    glutDisplayFunc (&drawGLScene);
    glutIdleFunc (&idleGLScene);
    glutReshapeFunc (&resizeGLScene);
    glutKeyboardFunc (&keyPressed);

    glClearColor (0.f, 0.f, 0.f, 1.f);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel (GL_SMOOTH);

    glGenTextures (1, &glRGBTex);
    glBindTexture (GL_TEXTURE_2D, glRGBTex);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}


// Displays the available controls 
void printInfo()
{
    std::cout << "\nAvailable Controls:\n";
    std::cout << "===================\n";
    std::cout << "Toggle Smoothing :  F\n";
    std::cout << "Tilt Kinect Up   :  W\n";
    std::cout << "Tilt Kinect Down :  S\n";
    std::cout << "Reset Tilt Angle :  R\n";
    std::cout << "Update LED State :  0-6\n";
    std::cout << "Quit             :  Q or Esc\n\n";
}


int main (int argc, char **argv)
{
    try
    {
        printInfo ();

        opencl = new Filter ();

        device = &freenect.createDevice<MyFreenectDevice> (0);
        device->startVideo ();

        initGL (argc, argv);
        glutMainLoop ();

        device->stopVideo ();
        delete opencl;

        return 0;
    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what () << " ("
                  << error.err ()  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}
