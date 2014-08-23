cmake_minimum_required ( VERSION 2.8.7 )
project ( kinectFilter )

list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package ( OpenCL REQUIRED )
find_package ( OpenCV REQUIRED )
find_package ( OpenGL REQUIRED )
find_package ( GLUT REQUIRED )
find_package ( Freenect REQUIRED )
find_package ( libusb-1.0 REQUIRED )

add_definitions ( -std=c++0x )

file ( COPY kernels.cl DESTINATION ${PROJECT_BINARY_DIR} )

include_directories ( ${FREENECT_INCLUDE_DIR}
                      ${LIBUSB_1_INCLUDE_DIRS}
                      ${OPENGL_INCLUDE_DIRS}
                      ${GLUT_INCLUDE_DIRS}
                      ${OpenCV_INCLUDE_DIR}
                      ${OPENCL_INCLUDE_DIR} )

add_executable ( kinectFilter kinectFilter.cpp )

target_link_libraries ( kinectFilter ${FREENECT_LIBRARIES}
                                     ${OPENGL_LIBRARIES}
                                     ${GLUT_LIBRARY}
                                     ${OpenCV_LIBS}
                                     ${OPENCL_LIBRARIES} )
