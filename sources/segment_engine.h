#ifndef __INC_SEGMENT_ENGINE_H
#define __INC_SEGMENT_ENGINE_H

#include "segment_context.h"

#include <future>


// 线程安全的上下文管理示例
struct Task {
    cv::Mat                         image_;
    std::promise<DetectResults>     results_;
};

// Instance Segmentation
class SegmentEngine {
public:
    explicit SegmentEngine();
    ~SegmentEngine();
    SegmentEngine(const SegmentEngine &) = delete;
    SegmentEngine &operator=(const SegmentEngine &) = delete;

    bool                        get_engine(const std::string &model_file);

    void                        InitWorkers(int32_t threadNum);
    void                        StopWorkers();

    DetectResults               RunSync(const cv::Mat &image);
    std::future<DetectResults>  RunAsync(const cv::Mat &image);

private:
    bool load_network_onnx(const std::string &model_file);
    bool load_network_engine(const std::string &engine_file);
    void inspect_engine(nvinfer1::INetworkDefinition *network) const;
    void get_model_dimensions();

protected:
    nvinfer1::IRuntime         *runtime_{nullptr};
    nvinfer1::ICudaEngine      *engine_{nullptr};

    const int32_t               input_index_{0};
    const int32_t               probe_index_{1};
    const int32_t               proto_index_{2};

private:
    // 异步推理资源
    void Run(SegmentContext &ctx);
    std::vector<SegmentContext> contexts_;
    std::vector<std::thread>    workers_;
    std::atomic<bool>           stopped_{false};
    std::queue<Task>            queue_;
    std::mutex                  mutex_;
    std::condition_variable     cv_;

    // 同步推理资源
    SegmentContext              context_;
};
#endif  //__INC_SEGMENT_ENGINE_H