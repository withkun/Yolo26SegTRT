cmake_minimum_required(VERSION 3.16)
project(YoloSegTRT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


################################################################################
# 使用当前日期设置版本号
if(NOT DEFINED TL_PROJECT_VERSION_MAJOR)
    string(TIMESTAMP DATE_TIME "%y-%m-%d %H:%M:%S")   # 25-08-03 08:49:05
    string(SUBSTRING ${DATE_TIME} 0 2 TL_PROJECT_VERSION_MAJOR)
    string(SUBSTRING ${DATE_TIME} 3 2 TL_PROJECT_VERSION_MINOR)
    string(SUBSTRING ${DATE_TIME} 6 2 TL_PROJECT_VERSION_PATCH)
    set(TL_PROJECT_VERSION_TWEAK 0)
endif()


################################################################################
set(OpenCV_DIR d:/3rd_party/opencv_4.13.0)
include(${OpenCV_DIR}/OpenCVConfig.cmake)

set(CUDA_PATH d:/3rd_party/cuda12.8_lib)
set(CUDA_SDK_ROOT_DIR ${CUDA_PATH})
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
set(CMAKE_CUDA_COMPILER ${CUDA_PATH}/bin/nvcc.exe)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#enable_language(CUDA)   #绝不可省略: 这个命令告诉系统要使用CUDA语言.

set(TENSOR_RT_ROOT d:/3rd_party/TensorRT-10.16.0.72)

################################################################################
add_subdirectory(deps/gflags-2.3.0)
add_subdirectory(deps/spdlog-1.17.0)

set(deps_libraries
    gflags spdlog::spdlog_header_only
)

################################################################################
add_executable(${PROJECT_NAME}
    sources/format_logger.h
    sources/nvidia_logger.h
    sources/nvidia_logger.cpp
    sources/segmentation.h
    sources/segmentation.cpp

    main.cpp
    application.rc
    versionInfo.rc
)
configure_file(version.rc.in ${PROJECT_BINARY_DIR}/versionInfo.rc)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sources
    ${CMAKE_CURRENT_SOURCE_DIR}/deps
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TENSOR_RT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    WIN64 _WIN32_WINNT=0x0601
    ${OPENCV_DEFINITION}
    SPDLOG_USE_STD_FORMAT
)

target_compile_options(${PROJECT_NAME} PRIVATE
    /MP /wd4267 /UTF-8 /bigobj /Ob1
)

target_link_options(${PROJECT_NAME} PRIVATE
    /DEBUG /NOLOGO
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${TlLib_Dir_TlAppV2}/lib/$<CONFIG>
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    ${TENSOR_RT_ROOT}/lib
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    cudart cublas
    nvinfer_10 nvonnxparser_10 nvinfer_plugin_10
    ${OpenCV_LIBS}
    ${deps_libraries}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
)