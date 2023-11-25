cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(torch_test)

set(CMAKE_PREFIX_PATH /home/kazeshini/Ctorch/libtorch/)

# use cudnn
set(CAFFE2_USE_CUDNN 1)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCES nn_main.cpp nn_main.h utils.h)

add_executable(${PROJECT_NAME} main.cpp ${SOURCES})

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
