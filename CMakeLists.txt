cmake_minimum_required(VERSION 3.18)
project(YoloSegTRT LANGUAGES C CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)


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
set(CUDA_PATH d:/3rd_party/cuda_v13.3_lib)
set(CUDA_SDK_ROOT_DIR ${CUDA_PATH})
set(CMAKE_CUDA_COMPILER ${CUDA_PATH}/bin/nvcc.exe)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

set(TENSOR_RT_ROOT d:/3rd_party/TensorRT-11.0.0.114)

set(OpenCV_DIR d:/3rd_party/opencv_5.0.0)
include(${OpenCV_DIR}/OpenCVConfig.cmake)


################################################################################
add_subdirectory(deps/gflags-2.3.0)
add_subdirectory(deps/spdlog-1.17.0)

set(deps_libraries
    gflags spdlog::spdlog_header_only
)


################################################################################
add_executable(${PROJECT_NAME}
    sources/format_logger.h
    sources/media_reader.h
    sources/media_reader.cpp
    sources/nvinfer_utils.h
    sources/nvinfer_utils.cpp
    sources/segment_utils.h
    sources/segment_utils.cpp
    sources/segment_context.h
    sources/segment_context.cpp
    sources/segment_engine.h
    sources/segment_engine.cpp

    main.cpp
    application.rc
    versionInfo.rc
)
configure_file(version.rc.in ${PROJECT_BINARY_DIR}/versionInfo.rc)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sources
    ${CMAKE_CURRENT_SOURCE_DIR}/deps
    ${CUDA_SDK_ROOT_DIR}/include
    ${ONNX_RUNTIME_ROOT}/include
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
    ${CUDA_SDK_ROOT_DIR}/lib/x64
    ${ONNX_RUNTIME_ROOT}/lib
    ${TENSOR_RT_ROOT}/lib
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    cudart cublas
    nvinfer_11 nvonnxparser_11 nvinfer_plugin_11
    ${OpenCV_LIBS}
    ${deps_libraries}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PDB_OUTPUT_DIRECTORY     "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build-relwithdebinfo"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build-relwithdebinfo"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build-relwithdebinfo"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
)