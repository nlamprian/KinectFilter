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
 * Filename: kinectFilter_gl_interop_vertex_buffer.cpp
 * File description: Example application making use of the OpenCL-OpenGL 
 *                   interoperability functionality. The use of vertex buffers 
 *                   is demonstrated. The OpenCL C++ API is utilized.
 *                   It creates a colored point cloud from the image 
 *                   and depth streams from Kinect.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>

#include <GL/glew.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <GL/glx.h>
#endif

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
const int gl_width = 640;
const int gl_height = 480;
int glWinId;

// Model parameters
int mouseX = -1, mouseY = -1;
float angleX = 0.f, angleY = 0.f;
float zoom = 1.f;

// OpenGL mem object parameters
void initGLObjects ();
GLuint glDepthBuf;
GLuint glRGBBuf;
bool color = true;

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
    Filter () : global { gl_width, gl_height }, rgb_norm (false)
    {
        // Image region for transfers
        region[0] = gl_width;
        region[1] = gl_height;
        region[2] = 1;

        // Image dimensions
        const int width = gl_width;
        const int height = gl_height;

        rgbBufferSize = 3 * sizeof (uint8_t) * width * height;
        rgbaBufferSize = 4 * sizeof (float) * width * height;
        depthBufferSize = sizeof (uint16_t) * width * height;

        // Get the list of platforms
        cl::Platform::get (&platforms);

        // Get the GPU devices in the first platform
        platforms[0].getDevices (CL_DEVICE_TYPE_GPU, &devices);

        // Detect OpenCL-OpenGL Interoperability
        checkCLGLInterop (devices[0]);

        // Create a context with CL-GL interop for the first device
        #if defined(_WIN32)
        cl_context_properties props[] = 
        {
            CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext (),
            CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC (),
            CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0]) (),
            0 
        };
        #elif defined(__APPLE__) || defined(__MACOSX)
        CGLContextObj kCGLContext = CGLGetCurrentContext ();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup (kCGLContext);
        cl_context_properties props[] = 
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties) kCGLShareGroup, 
            0 
        };
        #else
        cl_context_properties props[] = 
        {
            CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext (),
            CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay (),
            CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0]) (),
            0 
        };
        #endif
        // cl_device_id device;
        // clGetGLContextInfoKHR (props, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof (cl_device_id), &device, NULL);
        // devices.emplace_back (device);
        context = cl::Context (devices[0], props);

        // Create OpenGL memory objects
        initGLObjects ();

        // Create a command queue for the device
        queue = cl::CommandQueue (context, devices[0]);

        // Create a buffer instance for the source rgb image on the device
        bufferSourceRGB = cl::Buffer (context, CL_MEM_READ_ONLY, rgbBufferSize);

        // Create a buffer instance for the intermediate results on the device
        bufferInterRGBA = cl::Buffer (context, CL_MEM_READ_WRITE, rgbaBufferSize);

        // Create a buffer instance for the color image (shared with OpenGL) on the device
        bufferGLShared.emplace_back (context, CL_MEM_WRITE_ONLY, glRGBBuf);

        // Create a buffer instance for the source depth image on the device
        bufferSourceDepth = cl::Buffer (context, CL_MEM_READ_ONLY, depthBufferSize);

        // Create a buffer instance for the point cloud (shared with OpenGL) on the device
        bufferGLShared.emplace_back (context, CL_MEM_WRITE_ONLY, glDepthBuf);

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
        kernelRGBA = cl::Kernel (program, "rgb2rgba");
        kernelRGBNorm = cl::Kernel (program, "normalizeRGB");
        kernelDepthTo3D = cl::Kernel (program, "depthTo3D");

        // Set common kernel arguments
        kernelRGBA.setArg (0, bufferSourceRGB);
        kernelRGBA.setArg (2, height);
        kernelRGBA.setArg (3, width);

        kernelRGBNorm.setArg (0, bufferInterRGBA);
        kernelRGBNorm.setArg (1, bufferGLShared[0]);
        kernelRGBNorm.setArg (2, height);
        kernelRGBNorm.setArg (3, width);

        kernelDepthTo3D.setArg (0, bufferSourceDepth);
        kernelDepthTo3D.setArg (1, bufferGLShared[1]);
        kernelDepthTo3D.setArg (2, 595.f);
    }

    void processFrames (std::vector<uint8_t> &rgb, std::vector<uint16_t> &depth)
    {
        glFinish ();  // Wait for OpenGL pending operations on the buffers to finish

        // Take ownership of the OpenGL buffers
        queue.enqueueAcquireGLObjects ((std::vector<cl::Memory> *) &bufferGLShared);

        // Copy the source images to the device
        queue.enqueueWriteBuffer (bufferSourceRGB, CL_FALSE, 0, rgbBufferSize, rgb.data ());
        queue.enqueueWriteBuffer (bufferSourceDepth, CL_FALSE, 0, depthBufferSize, depth.data ());

        if (rgb_norm)
            kernelRGBA.setArg (1, bufferInterRGBA);
        else
            kernelRGBA.setArg (1, bufferGLShared[0]);

        // Restructure the data to include the A channel, and normalize 
        // (the final buffer object shared with OpenGL has to have RGBA 
        // channels, with float channel types and normalized values [0,1])
        queue.enqueueNDRangeKernel (kernelRGBA, cl::NullRange, global, cl::NullRange);

        // Perform RGB normalization        
        if (rgb_norm)
            queue.enqueueNDRangeKernel (kernelRGBNorm, cl::NullRange, global, cl::NullRange);

        // Transform depth image to 3D point cloud
        queue.enqueueNDRangeKernel (kernelDepthTo3D, cl::NullRange, global, cl::NullRange);

        // Give up ownership of the OpenGL buffers
        queue.enqueueReleaseGLObjects ((std::vector<cl::Memory> *) &bufferGLShared);

        queue.finish ();
    }

    // Returns the state of the flag for RGB normalization
    bool rgbNormalization ()
    {
        return rgb_norm;
    }

    // Toggles the flag for performing RGB normalization on the color image
    // Returns the new state of the flag
    bool toggleRGBNormalization ()
    {   
        rgb_norm = !rgb_norm;
        return rgb_norm;
    }

private:
    void checkCLGLInterop (cl::Device &device)
    {
        std::string exts;
        device.getInfo(CL_DEVICE_EXTENSIONS, &exts);

        #if defined(__APPLE__) || defined(__MACOSX)
        std::string glShare("cl_apple_gl_sharing");
        #else
        std::string glShare("cl_khr_gl_sharing");
        #endif

        if (exts.find (glShare) == std::string::npos)
        {
            std::cout << "OpenCL-OpenGL Interoperability"
                      << " is not supported on your device" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    // Source buffer parameters
    size_t rgbBufferSize, rgbaBufferSize, depthBufferSize;

    // Image transfer parameters
    cl::size_t<3> origin;
    cl::size_t<3> region;

    // Workspace dimensions
    cl::NDRange global;

    bool rgb_norm;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Buffer bufferSourceRGB, bufferInterRGBA;
    cl::Buffer bufferSourceDepth;
    std::vector<cl::BufferGL> bufferGLShared;
    cl::Program program;
    cl::Kernel kernelRGBA, kernelRGBNorm;
    cl::Kernel kernelDepthTo3D;
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
          depthBuffer (freenect_find_depth_mode (FREENECT_RESOLUTION_MEDIUM, 
                                                    FREENECT_DEPTH_REGISTERED).bytes / 2),
          newRGBFrame (false), newDepthFrame (false)
    {
        setDepthFormat(FREENECT_DEPTH_REGISTERED);
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

    // Delivers the latest Depth frame
    // Do not call directly, it's only used by the library
    void DepthCallback (void *depth, uint32_t timestamp)
    {
        Mutex::ScopedLock lock (depthMutex);

        std::copy (static_cast<uint16_t*> (depth), 
                   static_cast<uint16_t*> (depth) + getDepthBufferSize() / 2, 
                   depthBuffer.begin());
        newDepthFrame = true;
    }

    // Delivers the most recently received RGB frame
    bool getRGB (std::vector<uint8_t> &buffer)
    {
        Mutex::ScopedLock lock (rgbMutex);

        if (!newRGBFrame)
            return false;

        buffer.swap (rgbBuffer);
        newRGBFrame = false;

        return true;
    }

    // Delivers the most recently received Depth frame
    bool getDepth (std::vector<uint16_t> &buffer)
    {
        Mutex::ScopedLock lock (depthMutex);

        if (!newDepthFrame)
            return false;

        buffer.swap (depthBuffer);
        newDepthFrame = false;

        return true;
    }

private:
    Mutex rgbMutex, depthMutex;
    std::vector<uint8_t> rgbBuffer;
    std::vector<uint16_t> depthBuffer;
    bool newRGBFrame, newDepthFrame;
};


// If new frames are available, it processes them on the GPU
void updateFrames ()
{
    static std::vector<uint8_t> rgb (freenect_find_video_mode (
        FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes);
    static std::vector<uint16_t> depth (freenect_find_depth_mode (
        FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2);

    if (device->getRGB (rgb) && device->getDepth (depth))
    {
        opencl->processFrames (rgb, depth);    
    }
}


// Display callback for the window
void drawGLScene ()
{
    updateFrames ();

    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer (GL_ARRAY_BUFFER, glDepthBuf);
    glVertexPointer (4, GL_FLOAT, 0, NULL);
    glEnableClientState (GL_VERTEX_ARRAY);
    
    glBindBuffer (GL_ARRAY_BUFFER, glRGBBuf);
    glColorPointer (4, GL_FLOAT, 0, NULL);
    glEnableClientState (GL_COLOR_ARRAY);
    
    glDrawArrays (GL_POINTS, 0, gl_width * gl_height);

    glDisableClientState (GL_VERTEX_ARRAY);
    glDisableClientState (GL_COLOR_ARRAY);
    glBindBuffer (GL_ARRAY_BUFFER, 0);

    // Draw the world coordinate frame
    glLineWidth (2.f);
    glBegin (GL_LINES);
    glColor3ub (255, 0, 0);
    glVertex3i (  0, 0, 0);
    glVertex3i ( 50, 0, 0);

    glColor3ub (0, 255, 0);
    glVertex3i (0,   0, 0);
    glVertex3i (0,  50, 0);

    glColor3ub (0, 0, 255);
    glVertex3i (0, 0,   0);
    glVertex3i (0, 0,  50);
    glEnd ();

    // Position the camera
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();
    glScalef (zoom, zoom, 1);
    gluLookAt ( -7*angleX, -7*angleY, -1000.0,
                      0.0,       0.0,  2000.0,
                      0.0,      -1.0,     0.0 );

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
    gluPerspective (50.0, width / (float) height, 900.0, 11000.0);
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
        case 'C':
        case 'c':
            opencl->toggleRGBNormalization ();
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


// Mouse callack for the window
void mouseMoved (int x, int y)
{
    if (mouseX >= 0 && mouseY >= 0)
    {
        angleX += x - mouseX;
        angleY += y - mouseY;
    }

    mouseX = x;
    mouseY = y;
}


// Mouse buttons callback for the window
void mouseButtonPressed (int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        switch (button)
        {
            case GLUT_LEFT_BUTTON:
                mouseX = x;
                mouseY = y;
                break;
            case 3:  // Scroll Up
                zoom *= 1.2f;
                break;
            case 4:  // Scroll Down
                zoom /= 1.2f;
                break;
        }
    }
    else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
    {
        mouseX = -1;
        mouseY = -1;
    }
}


// Initializes GLUT
void initGL (int argc, char **argv)
{
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize (gl_width, gl_height);
    glutInitWindowPosition ((glutGet (GLUT_SCREEN_WIDTH) - gl_width) / 2,
                            (glutGet (GLUT_SCREEN_HEIGHT) - gl_height) / 2 - 70);
    glWinId = glutCreateWindow ("KinectFilter - CL-GL Interop - Vertex Buffers");

    glutDisplayFunc (&drawGLScene);
    glutIdleFunc (&idleGLScene);
    glutReshapeFunc (&resizeGLScene);
    glutKeyboardFunc (&keyPressed);
    glutMotionFunc (&mouseMoved);
    glutMouseFunc (&mouseButtonPressed);

    glewInit ();

    glClearColor (0.65f, 0.65f, 0.65f, 1.f);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_ALPHA_TEST);
    glAlphaFunc (GL_GREATER, 0.f);
    glEnable (GL_DEPTH_TEST);
    glShadeModel (GL_SMOOTH);
}


// Initializes OpenGL buffers
// Note: Call this after the OpenCL context has been created
void initGLObjects ()
{
    glGenBuffers (1, &glRGBBuf);
    glBindBuffer (GL_ARRAY_BUFFER, glRGBBuf);
    glBufferData (GL_ARRAY_BUFFER, 4 * sizeof (float) * gl_width * gl_height, NULL, GL_DYNAMIC_DRAW);
    glGenBuffers (1, &glDepthBuf);
    glBindBuffer (GL_ARRAY_BUFFER, glDepthBuf);
    glBufferData (GL_ARRAY_BUFFER, 4 * sizeof (float) * gl_width * gl_height, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer (GL_ARRAY_BUFFER, 0);
}


// Displays the available controls 
void printInfo()
{
    std::cout << "\nAvailable Controls:\n";
    std::cout << "===================\n";
    std::cout << "Rotate                   :  Mouse Left Button\n";
    std::cout << "Zoom In/Out              :  Mouse Wheel\n";
    std::cout << "Toggle RGB Normalization :  C\n";
    std::cout << "Tilt Kinect Up           :  W\n";
    std::cout << "Tilt Kinect Down         :  S\n";
    std::cout << "Reset Tilt Angle         :  R\n";
    std::cout << "Update LED State         :  0-6\n";
    std::cout << "Quit                     :  Q or Esc\n\n";
}


int main (int argc, char **argv)
{
    try
    {
        printInfo ();

        device = &freenect.createDevice<MyFreenectDevice> (0);
        device->startVideo ();
        device->startDepth ();

        initGL (argc, argv);

        // OpenCL environment must be created after the OpenGL environment 
        // has been initialized and before OpenGL starts rendering
        opencl = new Filter ();

        glutMainLoop ();

        device->stopVideo ();
        device->stopDepth ();
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
