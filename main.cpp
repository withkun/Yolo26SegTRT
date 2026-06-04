#include <iostream>
#include <filesystem>

#include "media_reader.h"
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
    SPDLOG_INFO("\xE7\xA8\x8B\xE5\xBA\x8F\xE5\x90\xAF\xE5\x8A\xA8 ..."); //程序启动UTF8编码，控制台不应显示乱码
    SPDLOG_INFO("Program started ...");
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
    const std::string image_file(FLAGS_image_file);
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);
    slogInit();
    SPDLOG_INFO("TensorRT YOLO26🚀实例分割");

    MediaReader image_generator(image_file);
    if (image_generator.empty()) {
        gflags::ShowUsageWithFlags(argv[0]);
        std::cerr << "===> TensorRT no image file found: " << image_file << std::endl;
        return -1;
    }
    SPDLOG_INFO("TensorRT total files: {}", image_generator.count());


    Segmentation segmentation;
    if (!segmentation.get_engine(model_file)) {
        gflags::ShowUsageWithFlags(argv[0]);
        std::cerr << "TensorRT no model file found" << std::endl;
        return -1;
    }

    cv::namedWindow("YOLO26+TensorRT", cv::WINDOW_FREERATIO | cv::WINDOW_GUI_EXPANDED);

    const auto frame_n = image_generator.count();
    const auto image_w = image_generator.width();
    const auto image_h = image_generator.height();
    SPDLOG_INFO("===> TensorRT image_sz: {}×{}, count: {}", image_h, image_w, frame_n);

    int64_t total_ms = 0;
    for (auto frame : image_generator.frames()) {
        const auto time1 = std::chrono::system_clock::now();

        DetectResults results = segmentation.RunSync(frame.image);

        DrawPred(frame.image, results);

        const auto time2 = std::chrono::system_clock::now();
        total_ms +=  std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
        SPDLOG_INFO("TensorRT instance segmentation: {}μs, instances: {}", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count(), results.size());
        cv::putText(frame.image, std::format("Frame: {:d}/{:d} fps: {:d} detect: {:d}", frame.index+1, image_generator.count(), (frame.index+1) * 1000 / total_ms, results.size()),
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::imshow("YOLO26+TensorRT", frame.image);

        if (!std::filesystem::exists(std::filesystem::path(FLAGS_output_dir))) {
            std::filesystem::create_directories(FLAGS_output_dir);
        }
        std::filesystem::path path = FLAGS_output_dir / std::filesystem::path(frame.source).filename();
        cv::imwrite(path.string(), frame.image);

        if (cv::waitKey(1) == VK_ESCAPE) {
            break;
        }
    }

    SPDLOG_INFO("TensorRT run finish. {}ms per image", total_ms / image_generator.count());
    cv::waitKey(0);
    cv::destroyAllWindows();

    spdlog::shutdown();
    return 0;
}