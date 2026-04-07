#ifndef INC_SEGMENTATION_H_
#define INC_SEGMENTATION_H_

#include "NvInfer.h"
#include "format_logger.h"
#include "nvidia_logger.h"

#include <future>


struct DetectResult {
    int32_t     id;         //结果类型Id
    float       confidence; //结果置信度
    cv::Rect2f  box;        //目标边界框
    cv::Mat     mask;       //目标掩码
};
using DetectResults = std::vector<DetectResult>;

inline int32_t calc_dims_size(const nvinfer1::Dims &v) {
    int32_t dim_size = 1;
    for (int i = 0; i < v.nbDims; ++i) {
        if (v.d[i] == 0 || v.d[i] == -1) {
            std::cerr << std::format("invalid dims for calc size: {}", v) << std::endl;
            throw std::runtime_error(std::format("invalid dims for calc size: {}", v));
        }
        dim_size *= static_cast<int32_t>(v.d[i]);
    }
    return dim_size;
}

inline std::string trim(const std::string &s) {
    if (s.empty()) return s;

    // 常见空白字符(C风格isspace范围)
    static constexpr std::string WHITESPACE = " \t\n\r\f\v";

    const size_t s_idx = s.find_first_not_of(WHITESPACE);
    if (s_idx == std::string::npos) return ""; // 全空白

    const size_t e_idx = s.find_last_not_of(WHITESPACE);
    return s.substr(s_idx, (e_idx - s_idx) + 1);
}

// 线程安全的上下文管理示例
struct Context {
    nvinfer1::IExecutionContext    *context_{nullptr};
    cudaStream_t                    stream_{nullptr};
    std::vector<void *>             buffers_;

    cv::Mat                         input_image_;
    cv::Mat                         output1_probes_;
    cv::Mat                         output2_protos_;
    float                           scale_xy_{0};
    float                           offset_x_{0};
    float                           offset_y_{0};

    int32_t                         image_h_{0};
    int32_t                         image_w_{0};
    DetectResults                   results_;
};

struct Task {
    cv::Mat                         image_;
    std::promise<DetectResults>     results_;
};

// Instance Segmentation
class Segmentation {
public:
    explicit Segmentation();
    ~Segmentation();
    Segmentation(const Segmentation &) = delete;
    Segmentation &operator=(const Segmentation &) = delete;

    bool                        get_engine(const std::string &model_file, const std::vector<int64_t> &image_dims);

    void                        InitWorkers(int32_t threadNum);
    void                        StopWorkers();

    DetectResults               RunSync(const cv::Mat &image);
    std::future<DetectResults>  RunAsync(const cv::Mat &image);

    int32_t                     chunk_size_{20};
    int32_t                     num_threads_{16};
    void                        MatrixMultiplyOptimize(Context &ctx, const cv::Mat &coefficients, const cv::Mat &prototypes) const;
    void                        MatrixMultiplyChunk(Context &ctx, const cv::Mat &coefficients, const cv::Mat &prototypes, const cv::Range &range) const;

protected:
    // 异步推理资源
    void Run(Context &ctx);
    std::vector<Context>        contexts_;
    std::vector<std::thread>    workers_;
    std::atomic<bool>           stopped_{false};
    std::queue<Task>            queue_;
    std::mutex                  mutex_;
    std::condition_variable     cv_;
    // 同步推理资源
    Context                     context_;

private:
    void inspect_engine(nvinfer1::INetworkDefinition *network) const;
    bool load_network_onnx(const std::string &model_file, const std::vector<int64_t> &image_dims);
    bool load_network_engine(const std::string &engine_file);
    void get_model_dimensions();

    void create_context(Context &ctx) const;
    void destroy_context(Context &ctx) const;

    void letterbox(const cv::Mat &image, Context &ctx) const;
    void inference(Context &ctx) const;
    void postprocess(const cv::Mat &image, Context &ctx, double conf_threshold, double mask_threshold) const;

protected:
    nvinfer1::IRuntime         *runtime_{nullptr};
    nvinfer1::ICudaEngine      *engine_{nullptr};

private:
    int32_t                     input_index_{0};
    int32_t                     output1_index_{1};
    int32_t                     output2_index_{2};

    nvinfer1::Dims              input_dims_{};    // nvinfer1::Dims{nbDims=4, d={1, 3, 1280, 1920, 0, 0, 0, 0}}
    nvinfer1::Dims              output1_dims_{};  // nvinfer1::Dims{nbDims=3, d={1, 300, 38, 0, 0, 0, 0, 0}}
    nvinfer1::Dims              output2_dims_{};  // nvinfer1::Dims{nbDims=4, d={1, 32, 320, 480, 0, 0, 0, 0}}

    // input: 预测图像张量(Image Tensor)
    int32_t                     INPUT_N_{1};
    int32_t                     INPUT_C_{3};
    int32_t                     INPUT_H_{1280};
    int32_t                     INPUT_W_{1920};
    // output0: 预测结果张量(Proposals Tensor)
    int32_t                     PROBES_N_{1};
    int32_t                     PROBES_H_{300};
    int32_t                     PROBES_W_{38};
    // output1: 掩膜原型张量(Mask Prototype Tensor)
    int32_t                     PROTOS_N_{1};
    int32_t                     PROTOS_C_{32};
    int32_t                     PROTOS_H_{320};
    int32_t                     PROTOS_W_{480};

    int32_t                     INPUT_SIZE_{INPUT_N_ * INPUT_C_ * INPUT_H_ * INPUT_W_};
    int32_t                     OUTPUT1_SIZE_{PROBES_N_ * PROBES_H_ * PROBES_W_};
    int32_t                     OUTPUT2_SIZE_{PROTOS_N_ * PROTOS_C_ * PROTOS_H_ * PROTOS_W_};
};
#endif  // INC_SEGMENTATION_H_