#include <iostream>
#include <filesystem>
#include <windows.h>

#include "segmentation.h"

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include "gflags/gflags.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"


DEFINE_bool(log_console, true, "show log console");
DEFINE_string(model_file, "best.onnx", "model file format of onnx or engine");
DEFINE_string(input_dims, "", "input image dimensions as NCHW(name:1,1,960,1280;name:1,1,960,1280)");
DEFINE_string(image_file, "images/*.png", "image file name or pattern");
DEFINE_string(output_dir, "results/", "output result directory");


// 初始化日志系统
static void slogInit() {
    std::vector<spdlog::sink_ptr> sinks;
    try {
        // 循环日志rotating_sink
        const std::string logFile("tlvision.log");
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(logFile, 50*1024*1024, 100, false));
        // 控制台日志console_sink
        if (FLAGS_log_console) {
            sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        }
    } catch (const spdlog::spdlog_ex &ex) {
        std::cerr << "Can not create log file: " << ex.what() << std::endl << "Program exit..." << std::endl;
        std::exit(-1);
    }

    // 创建异步日志
    spdlog::init_thread_pool(64, 1);
    const auto logger = std::make_shared<spdlog::async_logger>("TLA", sinks.begin(), sinks.end(), spdlog::thread_pool());
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    spdlog::flush_on(spdlog::level::info);
    spdlog::flush_every(std::chrono::milliseconds(10));
    spdlog::set_pattern("%^[%Y-%m-%dT%T.%f,%L,%t,%s:%#:%!]%$ %v");
    spdlog::set_level(spdlog::level::info);

    // 打印输出测试
    SPDLOG_INFO("程序启动 ...");
    SPDLOG_INFO("Program started ...");
    SPDLOG_INFO("\xE7\xA8\x8B\xE5\xBA\x8F\xE5\x90\xAF\xE5\x8A\xA8 ..."); //程序启动UTF8编码，控制台不应显示乱码
}

std::vector<cv::Scalar> label_colormap() {
    std::vector<cv::Scalar> colormap(256);
    for (int i = 0; i < 256; ++i) {
        // 提取标签i的8个二进制位
        const uint8_t b0 = (i >> 0) & 1;
        const uint8_t b1 = (i >> 1) & 1;
        const uint8_t b2 = (i >> 2) & 1;
        const uint8_t b3 = (i >> 3) & 1;
        const uint8_t b4 = (i >> 4) & 1;
        const uint8_t b5 = (i >> 5) & 1;
        const uint8_t b6 = (i >> 6) & 1;
        const uint8_t b7 = (i >> 7) & 1;
        // 合成RGB通道色彩值.
        const uint8_t r = (b0 << 7) | (b3 << 6) | (b6 << 5);
        const uint8_t g = (b1 << 7) | (b4 << 6) | (b7 << 5);
        const uint8_t b = (b2 << 7) | (b5 << 6);
        colormap[i] = cv::Scalar(r, g, b);
    }
    return colormap;
}
std::vector<cv::Scalar> LABEL_COLORMAP = label_colormap();

void DrawPred(cv::Mat &image, const std::vector<DetectResult> &results, const std::vector<cv::Scalar> &color_map) {
    const double fontScale = 0.5;
    const int32_t thickness = 1;
    const int32_t fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const int32_t lineType = cv::LINE_8;
    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }
    cv::Mat all_mask = image.clone();
    for (auto idx = 0; idx < results.size(); ++idx) {
        const auto &[id, confidence, box, mask] = results[idx];
        const auto color = color_map[idx % color_map.size()];
        cv::rectangle(image, box, color, thickness, lineType);

        // mask是一个与矩阵box大小相同的单通道二值掩码
        int32_t x1 = static_cast<int32_t>(box.x);
        int32_t y1 = static_cast<int32_t>(box.y);
        if (!mask.empty()) {
            all_mask(box).setTo(color, mask);
        }

        int32_t baseLine;
        std::string text = std::format("{}:{:.3f}", id, confidence);
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine);
        y1 = std::max(y1, textSize.height);
        cv::putText(image, text, cv::Point(x1, y1), fontFace, fontScale, color, thickness, lineType);
    }

    cv::addWeighted(image, 0.7, all_mask, 0.3, 0, image); //将mask加在原图上面
}

//https://blog.51cto.com/u_16099316/10633913
//https://developer.aliyun.com/article/1143198
//https://cloud.tencent.com/developer/article/2315250
int main(int argc, char **argv) {
    // YoloSegTRT.exe "C:/WORK/YoloSegTRT/yolo11n-seg.onnx" "C:/WORK/YoloSegTRT/dog.jpg"
    // YoloSegTRT.exe "C:/WORK/YoloSegTRT/yolov8n-seg.engine" "C:/WORK/YoloSegTRT/test3.mp4"
    // "d:/WORK/007.砂石骨料/stone/runs/segment/train2/weights/best.onnx" "d:/WORK/007.砂石骨料/20240513_all/2D相机采集图像/0_20210629_160940_0_12_STONE_E100_G100_H1624954180242_CTS1305342964_BF0.bmp"
    gflags::SetUsageMessage("Usage: example -model_file=name.{onnx, engine} -image_file=images/*.png -output_dir=results");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const std::string model_file(FLAGS_model_file);
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);
    slogInit();
    SPDLOG_INFO("TensorRT YOLO26🚀实例分割");

    std::vector<std::string> all_files;
    {
        std::string file_item;
        std::stringstream ss(FLAGS_image_file);
        while (std::getline(ss, file_item, ';')) {
            file_item = trim(file_item);
            if (file_item.empty()) continue;
            std::vector<std::string> im_files;
            cv::glob(file_item, im_files, false);
            all_files.insert(all_files.end(), im_files.begin(), im_files.end());
        }
    }

    if (all_files.empty()) {
        gflags::ShowUsageWithFlags(argv[0]);
        std::cerr << "TensorRT no image file found" << std::endl;
        return -1;
    }
    SPDLOG_INFO("TensorRT total files: {}", all_files.size());

    std::map<std::string, std::vector<int64_t>> dimensions;
    {
        std::string dims_item;
        std::stringstream ss(FLAGS_input_dims);     // name:NCHW;name:NCHW;name:NCHW;
        while (std::getline(ss, dims_item, ';')) {
            const auto pos = dims_item.find(':');
            if (pos == std::string::npos) {
                gflags::ShowUsageWithFlags(argv[0]);
                std::cerr << "input dimensions not accept: " << FLAGS_input_dims << std::endl;
                return -1;
            }
            std::string dim_item;
            std::vector<int64_t> value;
            std::string name = trim(dims_item.substr(0, pos));
            std::stringstream s1(dims_item.substr(pos + 1));     // NCHW
            while (std::getline(s1, dim_item, ',')) {
                dim_item = trim(dim_item);
                if (dim_item.empty()) continue;
                int32_t dim_value = std::stoi(dim_item);
                if (value.size() >= 2) {  // NCHW
                    dim_value = ((dim_value + 31) / 32) * 32;    // 向上取整.
                }
                value.push_back(dim_value);
            }
            if (name.empty() || value.empty()) {
                gflags::ShowUsageWithFlags(argv[0]);
                std::cerr << "input dimensions not accept: " << dims_item << std::endl;
                return -1;
            }

            dimensions[name] = value;
        }
    }

    Segmentation segmentation;
    if (!segmentation.get_engine(model_file, dimensions)) {
        gflags::ShowUsageWithFlags(argv[0]);
        std::cerr << "TensorRT no model file found" << std::endl;
        return -1;
    }

    int64_t total_ms = 0;
    for (int32_t index = 0; index < all_files.size(); ++index) {
        const auto time1 = std::chrono::system_clock::now();
        //std::cerr << "TensorRT read frame index: " << index << ", size: " << frame.cols << "x" << frame.rows << std::endl;
        cv::Mat image = cv::imread(all_files[index], cv::IMREAD_UNCHANGED);

        std::vector<DetectResult> results = segmentation.RunSync(image);
        DrawPred(image, results, LABEL_COLORMAP);

        const auto time2 = std::chrono::system_clock::now();
        total_ms +=  std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
        SPDLOG_INFO("TensorRT instance segmentation: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count());
        cv::putText(image, std::format("Frame: {:d}/{:d} fps: {:d} detect: {:d}", (index+1), all_files.size(), (index+1) * 1000000 / total_ms, results.size()),
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        if (!exists(std::filesystem::path(FLAGS_output_dir))) {
            std::filesystem::create_directories(FLAGS_output_dir);
        }

        std::filesystem::path path = FLAGS_output_dir / std::filesystem::path(all_files[index]).filename();
        cv::imwrite(path.string(), image);
        cv::imshow("YOLO26+TensorRT", image);
        if (cv::waitKey(1) == VK_ESCAPE) {
            break;
        }
    }

    SPDLOG_INFO("TensorRT run finish. {}ms per image", total_ms / all_files.size());
    cv::waitKey(0);
    cv::destroyAllWindows();
    spdlog::shutdown();
    return 0;
}