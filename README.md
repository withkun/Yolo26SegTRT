# Yolo26SegTRT
YOLO26 instance segmentation for TensorRT


# CUDA可下载最新版本:
https://developer.nvidia.com/cuda-downloads

https://developer.download.nvidia.com/compute/cuda/13.3.0/local_installers/cuda_13.3.0_windows.exe


# OpenCV可下载最新版本:
https://github.com/opencv/opencv

https://github.com/opencv/opencv/releases/download/5.0.0/opencv-5.0.0-windows.exe


# TensorRT根据本机CUDA下载对应版本:
https://developer.nvidia.com/tensorrt/download/10x

https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.0.0/zip/TensorRT-Enterprise-11.0.0.114-Windows-amd64-cuda-13.2-Release-external.zip


# 设置环境与执行:
SET PATH=d:\3rd_party\opencv_5.0.0\x64\vc16\bin;d:\3rd_party\cuda13.3_lib\bin;d:\3rd_party\TensorRT-11.0.0.114\bin;

YoloSegTRT.exe  -model_file=d:/WORK/YOLO26/runs/segment/train/weights/best_dyn.onnx  -input_dims=1,1,960,1280  -image_file=d:/WORK/测试图片/*.bmp;  -output_dir=runs/results > run.log 2>&1
