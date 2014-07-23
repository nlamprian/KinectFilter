/**
 * Name: KinectFilter
 * Author: Nick Lamprianidis { paign10.ln (at) gmail (dot) com }
 * Version: 1.0
 * Description: This is an example application putting together the following
 *              libraries: freenect, OpenCV, OpenCL, OpenGL. It uses a Kinect
 *              sensor as a camera. It applies a Laplacian filter on the video
 *              stream on the GPU with OpenCL, and then displays the processed
 *              stream on the screen.
 * Source: https://github.com/pAIgn10/KinectFilter
 * License: Copyright (c) 2014 Nick Lamprianidis
 *          This code is licensed under the GPL v2 license
 * Attribution: This application uses as a basis the "cppview" example of the
 *              libfreenect's C++ wrapper API.
 *              This example is part of the OpenKinect Project. www.openkinect.org
 *              Copyright (c) 2010 individual OpenKinect contributors
 *              See the CONTRIB file for details.
 *
 * Filename: kinectFilter.cpp
 * File description: Example application using the freenect and OpenCL libraries
 */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <pthread.h>

#include <libfreenect.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <CL/cl.h>

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

using namespace std;

class Filter
{
public:
    Filter() : origin{0,0,0}, region{640,480,1}, global{640,480}, local{16,16}
    {
        // Image dimensions
        int width = 640;
        int height = 480;

        // Laplacian kernel
        const float filter[] = { 1.0f, 1.0f, 1.0f,
                                 1.0f, -8.0f, 1.0f,
                                 1.0f, 1.0f, 1.0f };
        const int filterWidth = 3;
        const int filterSize = sizeof(float) * filterWidth * filterWidth;

        // Query for a plarform
        cl_platform_id platform;
        status = clGetPlatformIDs(1, &platform, NULL);
        chk("clGetPlatformIDs", status);

        // Query for a device
        cl_device_id device;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        chk("clGetDeviceIDs", status);

        // Create a context
        cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
        context = clCreateContext(cps, 1, &device, NULL, NULL, &status);
        chk("clCreateContext", status);

        // Create a command queue
        queue = clCreateCommandQueue(context, device, 0, &status);
        chk("clCreateCommandQueue", status);

        // Create a descriptor
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

        // Create an image format
        cl_image_format format;
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_UNSIGNED_INT8;

        // Create an image sampler
        cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status);
        chk("clCreateSampler", status);

        // Create an image object for the source image on the device
        bufferSourceImage = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, &status);
        chk("clCreateImage2D", status);

        // Create an image object for the output image on the device
        bufferOutputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &status);
        chk("clCreateImage2D", status);

        // Create a buffer for the filter on the device
        cl_mem bufferFilter = clCreateBuffer(context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
        chk("clCreateImage2D", status);

        // Copy the filter to the device
        status = clEnqueueWriteBuffer(queue, bufferFilter, CL_FALSE, 0, filterSize, filter, 0, NULL, NULL);
        chk("clEnqueueWriteBuffer", status);

        // Read the program source
        ifstream sourceFile("kernels.cl");
        string programCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
        const char* programSource = programCode.c_str();

        // Create the program
        program = clCreateProgramWithSource(context, 1, &programSource, NULL, &status);
        chk("clCreateProgramWithSource", status);

        // Compile the program
        status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (status)
        {
            char log[10240] = "";
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            cout << log;
            exit(-1);
        }

        // Create the kernel
        kernel_conv = clCreateKernel(program, "convolution", &status);
        chk("clCreateKernel", status);

        // Set the kernel arguments
        status = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), &bufferSourceImage);
        status |= clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), &bufferOutputImage);
        status |= clSetKernelArg(kernel_conv, 2, sizeof(int), &height);
        status |= clSetKernelArg(kernel_conv, 3, sizeof(int), &width);
        status |= clSetKernelArg(kernel_conv, 4, sizeof(cl_mem), &bufferFilter);
        status |= clSetKernelArg(kernel_conv, 5, sizeof(int), &filterWidth);
        status |= clSetKernelArg(kernel_conv, 6, sizeof(cl_sampler), &sampler);
        chk("clSetKernelArg", status);
    }

    void convolve(uint8_t *image) {
        // Copy the source image to the device
        status = clEnqueueWriteImage(queue, bufferSourceImage, CL_FALSE, origin, region, 0, 0, image, 0, NULL, NULL);
        chk("clEnqueueWriteImage", status);

        // Dispatch the kernel
        status = clEnqueueNDRangeKernel(queue, kernel_conv, 2, NULL, global, local, 0, NULL, NULL);
        chk("clEnqueueNDRangeKernel", status);

        // Read back the output image
        status = clEnqueueReadImage(queue, bufferOutputImage, CL_TRUE, origin, region, 0, 0, image, 0, NULL, NULL);
        chk("clEnqueueReadImage", status);
    }

private:
    void chk(const char* funcName, int errNum)
    {
        if (errNum != CL_SUCCESS)
        {
            cout << funcName << '(' << errNum << ')' << endl;
            exit(-1);
        }
    }

    // Image copy parameters
    size_t origin[3];
    size_t region[3];

    // Workgroup sizes
    size_t global[3];
    size_t local[3];

    cl_int status;
    cl_context context;
    cl_command_queue queue;
    cl_mem bufferSourceImage;
    cl_mem bufferOutputImage;
    cl_program program;
    cl_kernel kernel_conv;
};


class Mutex
{
public:
    Mutex()
    {
        pthread_mutex_init( &m_mutex, NULL );
    }

    void lock()
    {
        pthread_mutex_lock( &m_mutex );
    }

    void unlock()
    {
        pthread_mutex_unlock( &m_mutex );
    }

    class ScopedLock
    {
        Mutex & _mutex;
    public:
        ScopedLock(Mutex & mutex)
            : _mutex(mutex)
        {
            _mutex.lock();
        }

        ~ScopedLock()
        {
            _mutex.unlock();
        }
    };

private:
    pthread_mutex_t m_mutex;
};


class MyFreenectDevice : public Freenect::FreenectDevice
{
public:
    MyFreenectDevice(freenect_context *_ctx, int _index)
        : Freenect::FreenectDevice(_ctx, _index),
          m_buffer_video(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes),
          m_new_rgb_frame(false)
    {
        opencl = new Filter();
    }

    // Do not call directly, even in child
    void VideoCallback(void* _rgb, uint32_t timestamp)
    {
        Mutex::ScopedLock lock(m_rgb_mutex);
        uint8_t* rgb = static_cast<uint8_t*>(_rgb);
        copy(rgb, rgb+getVideoBufferSize(), m_buffer_video.begin());
        m_new_rgb_frame = true;
    };

    bool getRGB(vector<uint8_t> &buffer)
    {
        Mutex::ScopedLock lock(m_rgb_mutex);
        if (!m_new_rgb_frame)
            return false;

        // Create a vector of grayscale values
        cv::Mat rgb(480, 640, CV_8UC3, &m_buffer_video[0]);
        cv::Mat gray;
        cv::cvtColor(rgb, gray, CV_RGB2GRAY);
        uint8_t gray_buffer[640*480];
        copy(gray.datastart, gray.dataend+1, gray_buffer);

        // Apply the filter to the image
        opencl->convolve(gray_buffer);

        buffer.assign(gray_buffer, gray_buffer + sizeof(gray_buffer) / sizeof(gray_buffer[0]));
        m_new_rgb_frame = false;
        return true;
    }

private:
    Filter* opencl;
    vector<uint8_t> m_buffer_video;
    Mutex m_rgb_mutex;
    bool m_new_rgb_frame;
};


Freenect::Freenect freenect;
MyFreenectDevice* device;
freenect_video_format requested_format(FREENECT_VIDEO_RGB);

GLuint gl_rgb_tex;

double freenect_angle(0);
int got_frames(0), window(0);
int g_argc;
char **g_argv;


void DrawGLScene()
{
    static vector<uint8_t> image(640*480);

    device->updateState();
    printf("\r demanded tilt angle: %+4.2f device tilt angle: %+4.2f", freenect_angle, device->getState().getTiltDegs());
    fflush(stdout);

    device->getRGB(image);

    got_frames = 0;

    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, 1, 640, 480, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, &image[0]);

    glBegin(GL_TRIANGLE_FAN);
    glColor4f(255.0f, 255.0f, 255.0f, 255.0f);
    glTexCoord2f(0, 0); glVertex3f(0,0,0);
    glTexCoord2f(1, 0); glVertex3f(640,0,0);
    glTexCoord2f(1, 1); glVertex3f(640,480,0);
    glTexCoord2f(0, 1); glVertex3f(0,480,0);
    glEnd();

    glutSwapBuffers();
}

void keyPressed(unsigned char key, int x, int y)
{
    if (key == 27) {
        glutDestroyWindow(window);
    }
    if (key == '1') {
        device->setLed(LED_GREEN);
    }
    if (key == '2') {
        device->setLed(LED_RED);
    }
    if (key == '3') {
        device->setLed(LED_YELLOW);
    }
    if (key == '4') {
        device->setLed(LED_BLINK_GREEN);
    }
    if (key == '5') {
        // 5 is the same as 4
        device->setLed(LED_BLINK_GREEN);
    }
    if (key == '6') {
        device->setLed(LED_BLINK_RED_YELLOW);
    }
    if (key == '0') {
        device->setLed(LED_OFF);
    }

    if (key == 'w') {
        freenect_angle++;
        if (freenect_angle > 30) {
            freenect_angle = 30;
        }
    }
    if (key == 's' || key == 'd') {
        freenect_angle = 0;
    }
    if (key == 'x') {
        freenect_angle--;
        if (freenect_angle < -30) {
            freenect_angle = -30;
        }
    }
    if (key == 'e') {
        freenect_angle = 10;
    }
    if (key == 'c') {
        freenect_angle = -10;
    }
    device->setTiltDegrees(freenect_angle);
}

void ReSizeGLScene(int Width, int Height)
{
    glViewport(0,0,Width,Height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho (0, 640, 480, 0, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
}

void InitGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_SMOOTH);
    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void *gl_threadfunc(void *arg)
{
    printf("GL thread\n");

    glutInit(&g_argc, g_argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(0, 0);

    window = glutCreateWindow("KinectFilter");

    glutDisplayFunc(&DrawGLScene);
    glutIdleFunc(&DrawGLScene);
    glutReshapeFunc(&ReSizeGLScene);
    glutKeyboardFunc(&keyPressed);

    InitGL();

    glutMainLoop();

    return NULL;
}

int main(int argc, char **argv) {
    device = &freenect.createDevice<MyFreenectDevice>(0);
    device->startVideo();

    gl_threadfunc(NULL);

    device->stopVideo();

    return 0;
}
