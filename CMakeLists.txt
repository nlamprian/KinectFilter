cmake_minimum_required ( VERSION 2.8.12.2 )
project ( KinectFilter )

list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package ( OpenCL REQUIRED )
find_package ( OpenCV REQUIRED )
find_package ( OpenGL REQUIRED )
find_package ( GLEW REQUIRED )
find_package ( GLUT REQUIRED )
find_package ( Threads REQUIRED )
find_package ( libusb-1.0 REQUIRED )
find_package ( Freenect REQUIRED )

add_definitions ( -std=c++0x )

# set ( WARNINGS "-Wall -Wextra" )
# set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNINGS}" )

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )

add_custom_target ( kernels ALL )
add_custom_command ( 
    TARGET kernels POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/kernels/kernels.cl ${PROJECT_BINARY_DIR}/kernels/kernels.cl
)

include_directories ( 
    ${LIBUSB_1_INCLUDE_DIRS}
    ${FREENECT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${OPENCL_INCLUDE_DIR} 
    ${GLEW_INCLUDE_DIRS}
)

add_executable ( 
    kinectFilter_clc 
    src/kinectFilter_clc.cpp 
)

add_executable ( 
    kinectFilter_clc++ 
    src/kinectFilter_clc++.cpp 
)

add_executable ( 
    kinectFilter_gl_interop_texture 
    src/kinectFilter_gl_interop_texture.cpp 
)

add_executable ( 
    kinectFilter_gl_interop_vertex_buffer
    src/kinectFilter_gl_interop_vertex_buffer.cpp 
)

set ( 
    LINK_LIBS 
    ${CMAKE_THREAD_LIBS_INIT}
    ${FREENECT_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    ${OpenCV_LIBS}
    ${OPENCL_LIBRARIES} 
    ${GLEW_LIBRARIES}
)

target_link_libraries ( 
    kinectFilter_clc 
    ${LINK_LIBS}
)

target_link_libraries ( 
    kinectFilter_clc++ 
    ${LINK_LIBS}
)

target_link_libraries ( 
    kinectFilter_gl_interop_texture 
    ${LINK_LIBS}
)

target_link_libraries ( 
    kinectFilter_gl_interop_vertex_buffer 
    ${LINK_LIBS}
)
