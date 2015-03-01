/**
 * Name: KinectFilter
 * Author: Nick Lamprianidis { paign10.ln (at) gmail (dot) com }
 * Version: 2.0
 * Description: This project brings together the following libraries: 
 *              libfreenect, OpenCV, OpenGL, OpenCL. There is a number of 
 *              applications that use a Kinect sensor as a camera, process the 
 *              data stream from Kinect on the GPU with OpenCL, and display the 
 *              processed stream in a graphical window.
 * Source: https://github.com/pAIgn10/KinectFilter
 * License: Copyright (c) 2014-2015 Nick Lamprianidis
 *          This code is licensed under the GPL v2 license
 * Attribution: This application was based on the "cppview" example of the
 *              libfreenect's C++ wrapper API.
 *              That example is part of the OpenKinect Project. www.openkinect.org
 *              Copyright (c) 2010 individual OpenKinect contributors
 *              See the CONTRIB file for details.
 *
 * Filename: kinectFilter_clc.cpp
 * File description: Example application utilizing the OpenCL C API. 
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

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#include <GLUT/glut.h>
#else
#include <CL/cl.h>
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
    Filter () : origin { 0, 0, 0 }, region { gl_win_width, gl_win_height, 1 }, 
                global { gl_win_width, gl_win_height }, smoothed (true)
    {
        // Image dimensions
        const int width = gl_win_width;
        const int height = gl_win_height;

        // Applying multiple times a box filter, approximates a Gaussian filter
        const float box_filter[] = { 0.125f, 0.125f, 0.125f,
                                     0.125f, 0.125f, 0.125f,
                                     0.125f, 0.125f, 0.125f };
        const float laplacian_filter[] = { 1.f,  1.f, 1.f,
                                           1.f, -8.f, 1.f,
                                           1.f,  1.f, 1.f };
        const int filterWidth = 3;
        const int filterSize = filterWidth * filterWidth * sizeof (float);

        // Query for a plarform
        cl_platform_id platform;
        status = clGetPlatformIDs (1, &platform, NULL);
        chk ("clGetPlatformIDs", status);

        // Query for a device
        cl_device_id device;
        status = clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        chk ("clGetDeviceIDs", status);

        // Create a context
        cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0 };
        context = clCreateContext (cps, 1, &device, NULL, NULL, &status);
        chk ("clCreateContext", status);

        // Create a command queue
        queue = clCreateCommandQueue (context, device, 0, &status);
        chk ("clCreateCommandQueue", status);

        // Create image descriptor
        cl_image_desc desc;
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = width;
        desc.image_height = height;
        desc.image_depth = 0;
        desc.image_array_size = 0;
        desc.image_row_pitch = 0;
        desc.image_slice_pitch = 0;
        desc.num_mip_levels = 0;
        desc.num_samples = 0;
        desc.buffer = NULL;

        // Create image format
        cl_image_format format;
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_UNSIGNED_INT8;

        // Create an image sampler
        sampler = clCreateSampler (context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status);
        chk ("clCreateSampler", status);

        // Create an image object for the source image on the device
        bufferSourceImage = clCreateImage (context, CL_MEM_READ_ONLY, &format, &desc, NULL, &status);
        chk ("clCreateImage2D", status);

        // Create image objects for the intermediate results on the device
        bufferInterImage1 = clCreateImage (context, CL_MEM_READ_WRITE, &format, &desc, NULL, &status);
        chk ("clCreateImage2D", status);
        bufferInterImage2 = clCreateImage (context, CL_MEM_READ_WRITE, &format, &desc, NULL, &status);
        chk ("clCreateImage2D", status);

        // Create an image object for the output image on the device
        bufferOutputImage = clCreateImage (context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &status);
        chk ("clCreateImage2D", status);

        // Create buffers for the filters on the device
        bufferBoxFilter = clCreateBuffer (context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
        chk ("clCreateImage2D", status);
        bufferLaplacianFilter = clCreateBuffer (context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
        chk ("clCreateImage2D", status);

        // Copy the filters to the device
        status = clEnqueueWriteBuffer (queue, bufferBoxFilter, CL_FALSE, 0, filterSize, box_filter, 0, NULL, NULL);
        chk ("clEnqueueWriteBuffer", status);
        status = clEnqueueWriteBuffer (queue, bufferLaplacianFilter, CL_FALSE, 0, filterSize, laplacian_filter, 0, NULL, NULL);
        chk ("clEnqueueWriteBuffer", status);

        // Read the program source
        std::ifstream sourceFile ("kernels/kernels.cl");
        std::string programCode (std::istreambuf_iterator<char> (sourceFile), (std::istreambuf_iterator<char> ()));
        const char* programSource = programCode.c_str ();

        // Create program
        program = clCreateProgramWithSource (context, 1, &programSource, NULL, &status);
        chk ("clCreateProgramWithSource", status);

        // Compile program
        status = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
        if (status)
        {
            char log[10240] = "";
            clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, sizeof (log), log, NULL);
            std::cerr << log << std::endl;
            exit (EXIT_FAILURE);
        }

        // Create kernel
        kernelConv = clCreateKernel (program, "convolution", &status);
        chk ("clCreateKernel", status);

        // Set common kernel arguments
        status = clSetKernelArg (kernelConv, 2, sizeof (int), &height);
        status |= clSetKernelArg (kernelConv, 3, sizeof (int), &width);
        status |= clSetKernelArg (kernelConv, 5, sizeof (int), &filterWidth);
        status |= clSetKernelArg (kernelConv, 6, sizeof (cl_sampler), &sampler);
        chk ("clSetKernelArg", status);
    }

    void convolve (std::vector<uint8_t> &image)
    {
        // Copy the source image to the device
        status = clEnqueueWriteImage (queue, bufferSourceImage, CL_FALSE, origin, region, 0, 0, image.data (), 0, NULL, NULL);
        chk ("clEnqueueWriteImage", status);

        if (smoothed)
        {
            status = clSetKernelArg (kernelConv, 0, sizeof (cl_mem), &bufferSourceImage);
            status |= clSetKernelArg (kernelConv, 1, sizeof (cl_mem), &bufferInterImage1);
            status |= clSetKernelArg (kernelConv, 4, sizeof (cl_mem), &bufferBoxFilter);
            chk ("clSetKernelArg", status);

            // Apply the first box firter
            status = clEnqueueNDRangeKernel (queue, kernelConv, 2, NULL, global, NULL, 0, NULL, NULL);
            chk ("clEnqueueNDRangeKernel", status);

            status = clSetKernelArg (kernelConv, 0, sizeof (cl_mem), &bufferInterImage1);
            status |= clSetKernelArg (kernelConv, 1, sizeof (cl_mem), &bufferInterImage2);
            chk ("clSetKernelArg", status);

            // Apply the second box firter
            status = clEnqueueNDRangeKernel (queue, kernelConv, 2, NULL, global, NULL, 0, NULL, NULL);
            chk ("clEnqueueNDRangeKernel", status);

            status = clSetKernelArg (kernelConv, 0, sizeof (cl_mem), &bufferInterImage2);
            chk ("clSetKernelArg", status);
        }
        else
        {
            status = clSetKernelArg (kernelConv, 0, sizeof (cl_mem), &bufferSourceImage);
            chk ("clSetKernelArg", status);
        }

        status = clSetKernelArg (kernelConv, 1, sizeof (cl_mem), &bufferOutputImage);
        status |= clSetKernelArg (kernelConv, 4, sizeof (cl_mem), &bufferLaplacianFilter);
        chk ("clSetKernelArg", status);

        // Perform the edge detection
        status = clEnqueueNDRangeKernel (queue, kernelConv, 2, NULL, global, NULL, 0, NULL, NULL);
        chk ("clEnqueueNDRangeKernel", status);

        // Read back the output image
        status = clEnqueueReadImage (queue, bufferOutputImage, CL_TRUE, origin, region, 0, 0, image.data (), 0, NULL, NULL);
        chk ("clEnqueueReadImage", status);
    }

    ~Filter ()
    {
        clReleaseKernel (kernelConv);
        clReleaseProgram (program);
        clReleaseMemObject (bufferSourceImage);
        clReleaseMemObject (bufferInterImage1);
        clReleaseMemObject (bufferInterImage2);
        clReleaseMemObject (bufferOutputImage);
        clReleaseMemObject (bufferBoxFilter);
        clReleaseMemObject (bufferLaplacianFilter);
        clReleaseSampler (sampler);
        clReleaseCommandQueue (queue);
        clReleaseContext (context);
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
    void chk (const char* funcName, int errNum)
    {
        if (errNum != CL_SUCCESS)
        {
            std::cerr << funcName << " (" << errNum << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }

    // Image transfer parameters
    size_t origin[3];
    size_t region[3];

    // Workspace dimensions
    size_t global[2];

    bool smoothed;

    cl_int status;
    cl_context context;
    cl_command_queue queue;
    cl_sampler sampler;
    cl_mem bufferBoxFilter, bufferLaplacianFilter;
    cl_mem bufferSourceImage, bufferOutputImage;
    cl_mem bufferInterImage1, bufferInterImage2;
    cl_program program;
    cl_kernel kernelConv;
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
    glWinId = glutCreateWindow ("KinectFilter - OpenCL C API");

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
