# Yolo26SegTRT
YOLO26 instance segmentation for TensorRT


# OpenCV可下载最新版本:
https://github.com/opencv/opencv

https://github.com/opencv/opencv/releases/download/4.13.0/opencv-4.13.0-windows.exe

# TensorRT根据本机CUDA下载对应版本:
https://developer.nvidia.com/tensorrt/download/10x

https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.16.0/zip/TensorRT-10.16.0.72.Windows.amd64.cuda-12.9.zip


# 设置环境与执行:
SET PATH=d:\3rd_party\opencv_4.13.0\x64\vc16\bin;d:\3rd_party\cuda12.8_lib\bin;d:\3rd_party\TensorRT-10.16.0.72\bin;
YoloSegTRT.exe  -model_file=d:/WORK/YOLO26/runs/segment/train/weights/best_dyn.onnx  -input_dims=1,1,960,1280  -image_file=d:/WORK/测试图片/*.bmp;  -output_dir=runs/results > run.log 2>&1
