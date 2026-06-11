#include "segment_context.h"
#include "cuda_runtime.h"


SegmentContext::SegmentContext(nvinfer1::ICudaEngine *engine) {
    create_context(engine);
}

SegmentContext::~SegmentContext() {
    destroy_context();
}

SegmentContext::SegmentContext(SegmentContext &&rsh) noexcept {
    *this = std::move(rsh);
    rsh.stream_ = nullptr;
}

SegmentContext &SegmentContext::operator=(SegmentContext &&rsh) noexcept {
    this->stream_           = rsh.stream_;
    this->context_          = std::move(rsh.context_);

    this->input_dims_       = rsh.input_dims_;
    this->probe_dims_       = rsh.probe_dims_;
    this->proto_dims_       = rsh.proto_dims_;

    this->d_image_          = rsh.d_image_;
    this->d_proposals_      = rsh.d_proposals_;
    this->d_prototypes_     = rsh.d_prototypes_;
    this->h_image_          = rsh.h_image_;
    this->h_proposals_      = rsh.h_proposals_;
    this->h_prototypes_     = rsh.h_prototypes_;
    this->results_          = rsh.results_;

    // Input Tensor:            nvinfer1::Dims{nbDims=4, d={1, 1, 1024, 1024, 0, 0, 0, 0}}
    this->INPUT_N_          = rsh.INPUT_N_;
    this->INPUT_C_          = rsh.INPUT_C_;
    this->INPUT_H_          = rsh.INPUT_H_;
    this->INPUT_W_          = rsh.INPUT_W_;
    // Proposals Tensor:        nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}}
    this->PROBE_N_          = rsh.PROBE_N_;
    this->PROBE_H_          = rsh.PROBE_H_;
    this->PROBE_W_          = rsh.PROBE_W_;
    // Prototypes Tensor:       nvinfer1::Dims{nbDims=4, d={1, 32, 256, 256, 0, 0, 0, 0}}
    this->PROTO_N_          = rsh.PROTO_N_;
    this->PROTO_C_          = rsh.PROTO_C_;
    this->PROTO_H_          = rsh.PROTO_H_;
    this->PROTO_W_          = rsh.PROTO_W_;

    this->INPUT_SIZE_       = rsh.INPUT_SIZE_;
    this->PROBE_SIZE_       = rsh.PROBE_SIZE_;
    this->PROTO_SIZE_       = rsh.PROTO_SIZE_;

    this->image_h_          = rsh.image_h_;
    this->image_w_          = rsh.image_w_;
    this->padded_x_         = rsh.padded_x_;
    this->padded_y_         = rsh.padded_y_;
    this->scale_xy_         = rsh.scale_xy_;
    this->sampling_         = rsh.sampling_;


    rsh.stream_ = nullptr;
    return *this;
}

void SegmentContext::create_context(nvinfer1::ICudaEngine *engine) {
    const auto time_point1 = std::chrono::system_clock::now();
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // 计算输入输出尺寸.
    std::vector<const char *> tensor_names{ INPUT_BLOB_NAME, OUTPUT1_BLOB_NAME, OUTPUT2_BLOB_NAME };
    context_->inferShapes(tensor_names.size(), tensor_names.data());

    // 获取模型输入尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 1, 1024, 1024, 0, 0, 0, 0}})
    input_dims_ = context_->getTensorShape(INPUT_BLOB_NAME);
    SPDLOG_INFO("TensorRT input Dimensions: {}", input_dims_);
    INPUT_N_ = static_cast<int32_t>(input_dims_.d[0]);      // 1
    INPUT_C_ = static_cast<int32_t>(input_dims_.d[1]);      // 1
    INPUT_H_ = static_cast<int32_t>(input_dims_.d[2]);      // 1024
    INPUT_W_ = static_cast<int32_t>(input_dims_.d[3]);      // 1024
    INPUT_SIZE_ = DimsInBytes(input_dims_);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}})
    probe_dims_ = context_->getTensorShape(OUTPUT1_BLOB_NAME);
    SPDLOG_INFO("TensorRT output1 Dimensions: {}", probe_dims_);
    PROBE_N_ = static_cast<int32_t>(probe_dims_.d[0]);      // 1
    PROBE_H_ = static_cast<int32_t>(probe_dims_.d[1]);      // 960
    PROBE_W_ = static_cast<int32_t>(probe_dims_.d[2]);      // 38
    PROBE_SIZE_ = DimsInBytes(probe_dims_);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 32, 256, 256, 0, 0, 0, 0}})
    proto_dims_ = context_->getTensorShape(OUTPUT2_BLOB_NAME);
    SPDLOG_INFO("TensorRT output2 Dimensions: {}", proto_dims_);
    PROTO_N_ = static_cast<int32_t>(proto_dims_.d[0]);      // 1
    PROTO_C_ = static_cast<int32_t>(proto_dims_.d[1]);      // 32
    PROTO_H_ = static_cast<int32_t>(proto_dims_.d[2]);      // 256
    PROTO_W_ = static_cast<int32_t>(proto_dims_.d[3]);      // 256
    PROTO_SIZE_ = DimsInBytes(proto_dims_);

    // 分配GPU内存, 绑定数据缓冲区.
    cudaStreamCreate(&stream_);
    cudaMalloc(&d_image_,      INPUT_SIZE_ * sizeof(float));
    cudaMalloc(&d_proposals_,  PROBE_SIZE_ * sizeof(float));
    cudaMalloc(&d_prototypes_, PROTO_SIZE_ * sizeof(float));
    context_->setTensorAddress(INPUT_BLOB_NAME,   d_image_);
    context_->setTensorAddress(OUTPUT1_BLOB_NAME, d_proposals_);
    context_->setTensorAddress(OUTPUT2_BLOB_NAME, d_prototypes_);
    h_proposals_.resize(PROBE_SIZE_);
    h_prototypes_.resize(PROTO_SIZE_);
    SPDLOG_INFO("TensorRT Init, Context: {}:{} {}:{}:{}", static_cast<void *>(this), static_cast<void *>(context_.get()), static_cast<void *>(d_image_), static_cast<void *>(d_proposals_), static_cast<void *>(d_prototypes_));
}

void SegmentContext::destroy_context() {
    if (stream_ == nullptr) {
        return;
    }

    context_->setTensorAddress(INPUT_BLOB_NAME,   nullptr);
    context_->setTensorAddress(OUTPUT1_BLOB_NAME, nullptr);
    context_->setTensorAddress(OUTPUT2_BLOB_NAME, nullptr);
    cudaFree(d_image_);
    cudaFree(d_proposals_);
    cudaFree(d_prototypes_);

    context_.reset();
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
}

DetectResults SegmentContext::RunSync(const cv::Mat &image) {
    letterbox(image);

    inference();

    postprocess(image, CONF_THRESHOLD, MASK_THRESHOLD);

    return results_;
}

void SegmentContext::letterbox(const cv::Mat &image) {
    image_h_ = image.rows;
    image_w_ = image.cols;
    SPDLOG_INFO("TensorRT letterbox image size: {}", image.size);

    // 图像预处理方法: 计算缩放比例, 取其中较小的一侧以保持原图的宽高比.
    scale_xy_ = std::min(1.0f * INPUT_H_ / image.rows, 1.0f * INPUT_W_ / image.cols);
    const auto scaled_h = static_cast<int32_t>(image.rows * scale_xy_);
    const auto scaled_w = static_cast<int32_t>(image.cols * scale_xy_);
    padded_y_ = INPUT_H_ - scaled_h;
    padded_x_ = INPUT_W_ - scaled_w;
    SPDLOG_INFO("TensorRT letterbox scale_xy: {} offset_x: {} offset_y: {} image: {}", scale_xy_, padded_x_, padded_y_, image.ptr<void>());

    // cv::INTER_AREA适用于缩小图像, 放大图像可能得到非预期的结果.
    // 如果需要放大图像, 应该使用cv::INTER_LINEAR或cv::INTER_CUBIC.
    cv::Mat final;
    if (INPUT_H_ == image_h_ && INPUT_W_ == image_w_) {
        // 不要缩放, 不要填充
        final = image;
    } else if (INPUT_H_ == image_h_ || INPUT_W_ == image_w_) {
        // 不要缩放, 需要填充
        cv::copyMakeBorder(image, final, 0, padded_y_, 0, padded_x_, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    } else {
        // 需要缩放, 需要填充
        cv::resize(image, final, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(final, final, 0, padded_y_, 0, padded_x_, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    }
    SPDLOG_INFO("TensorRT letterbox scale image size: {}", final.size);

    // image: 输入图像, 灰度图或三通道图(一般为BGR).
    // blob: 输出4维矩阵, 符合模型输入的NCHW格式. [1, C, H, W]
    // scalefactor: 缩放因子, 图像像素值的缩放比例; 图像像素减去平均值之后, 再进行缩放, 默认值是1.
    // size: 目标尺寸, 模型输入的图片尺寸.
    // mean: 图像要减去均值, 如果需要对BGR图片的三个通道分别减去不同的值, 可以使用3个值; 如果三通道图像只有1个值, 那么三个通道都减去相同的值.
    // swapRB: OpenCV中图片通道顺序是BGR, 但是假设输入顺序是RGB, 处理时可以同步转换为RGB格式, 那么就要使swapRB=true.
    // crop: 是否裁剪, 调整尺寸时是保持比例并裁剪(非拉伸), 如果crop裁剪为true, 则调整输入图像的大小, 使调整大小后的一侧等于相应的尺寸, 另一侧等于或大于, 然后从中心进行裁剪; 如果crop裁剪为false, 则直接调整大小而不进行裁剪并保留纵横比.
    // ddepth: 输出数据类型, 通常为CV_32F或CV_8U.
    cv::dnn::blobFromImage(final, h_image_, 1.0/255.0, cv::Size(INPUT_W_, INPUT_H_), cv::Scalar(), true, false, CV_32F);
    SPDLOG_INFO("TensorRT letterbox final image size: {} blob: {}", final.size, h_image_.ptr<void>());
}

void SegmentContext::inference() {
    SPDLOG_INFO("TensorRT inference enter");
    // 异步流拷贝输入数据
    cudaMemcpyAsync(d_image_, h_image_.ptr<float>(0), INPUT_SIZE_ * sizeof(float), cudaMemcpyHostToDevice, stream_);

    // 异步流提交推理任务
    context_->enqueueV3(stream_);

    // 异步流拷贝输出数据
    cudaMemcpyAsync(h_proposals_.data(), d_proposals_, PROBE_SIZE_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_prototypes_.data(), d_prototypes_, PROTO_SIZE_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    // 流同步等待处理完成
    cudaStreamSynchronize(stream_);
    SPDLOG_INFO("TensorRT inference leave");
}

void SegmentContext::postprocess(const cv::Mat &image, const float conf_threshold, const float mask_threshold) {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // YOLO26模型输出:
    // output0输出为预测结果张量, 其维度是(1,960,38)  NHW: (x1,y1,x2,y2, score, class_id, mask1,mask2,...,mask32)
    // 38为4+1+1+32, 4为box的[x1,y1,x2,y2], 1个类别置信度, 1个类别标识, 32个掩膜系数(mask coefficients), 960个预测候选结果.
    // output1输出为掩膜原型张量, 其维度是(1,32,320,480), 32对应掩膜系数. 掩膜原型(320×480)需先上采样至输入图像尺寸(1280×1920).
    // 模型在推理时, 会根据每个检测框的32个掩膜系数, 对这些掩膜原型进行加权求和, 从而生成每个目标的掩膜.
    // YOLO26预测结果数据结构示例:
    // (nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}})   // 960行, 38列
    // 38为4+1+1+32, 4个边界框, 1个类别置信度, 1个类别标识, 32是掩膜系数
    // box1{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box2{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box3{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box4{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box5{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box960
    SPDLOG_INFO(std::format("TensorRT offset_x:{} offset_y:{} scale_xy:{}", padded_x_, padded_y_, scale_xy_));
    cv::Mat coefficients;  // 掩膜权重
    coefficients.reserve(PROBE_H_);
    results_.clear();
    results_.reserve(PROBE_H_);

    // 根据置信度过滤(Proposals Tensor)
    const float *proposals = h_proposals_.data();
    for (int32_t row = 0; row < PROBE_H_; ++row) {
        if (proposals[row * PROBE_W_ + 4] < conf_threshold) {
            continue;
        }
        results_.push_back({});
        DetectResult &result = results_.back();
        result.score = proposals[row * PROBE_W_ + 4];
        result.label = proposals[row * PROBE_W_ + 5];

        const cv::Mat probe = cv::Mat(1, 32, CV_32FC1, (void *)(proposals + row * PROBE_W_ + 6)).clone();
        coefficients.push_back(probe);   // HW (1,32)

        // 边界框格式: xyxy
        const auto x1 = proposals[row * PROBE_W_ + 0];
        const auto y1 = proposals[row * PROBE_W_ + 1];
        const auto x2 = proposals[row * PROBE_W_ + 2];
        const auto y2 = proposals[row * PROBE_W_ + 3];
        result.bbox = cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
    }
    SPDLOG_INFO("TensorRT probe size after conf_threshold: {}", results_.size());

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 掩膜原型处理 (Mask Prototype Tensor)
    const cv::Mat prototypes = cv::Mat(1, 1, CV_32FC1, h_prototypes_.data()) ;
    // 多线程矩阵乘: coefficients @ prototypes ^ T
    MatrixMultiplyOptimize(coefficients, prototypes, mask_threshold);
    SPDLOG_INFO("TensorRT after postprocess: {}", results_.size());
}

// 分块矩阵乘法核心实现
// coefficients: 掩码系数矩阵 (n x k)
// prototypes: 原型掩码矩阵 (k x (h*w))
void SegmentContext::MatrixMultiplyChunk(const cv::Mat &coefficients, const cv::Mat &prototypes,
                                         const cv::Range &range, const float mask_threshold) {
    // 计算原型掩码的缩放比例
    const float scale_x = 1.0 * PROTO_W_ / INPUT_W_;
    const float scale_y = 1.0 * PROTO_H_ / INPUT_H_;

    const auto t0 = cv::getTickCount();
    const auto path = std::format("{}", t0);
    //std::filesystem::create_directory(path);

    // 对每个检测目标执行矩阵乘法运算
    for (int32_t n = range.start; n < range.end && n < coefficients.rows; ++n) {
        auto &box = results_[n].bbox;
        const auto tl = box.tl();
        const auto br = box.br();
        // 将边界框坐标转换到原型掩码空间
        const auto x1 = std::round(std::max(0.0f, std::min(tl.x * scale_x, 1.0f * PROTO_W_)));
        const auto y1 = std::round(std::max(0.0f, std::min(tl.y * scale_y, 1.0f * PROTO_H_)));
        const auto x2 = std::round(std::max(0.0f, std::min(br.x * scale_x, 1.0f * PROTO_W_)));
        const auto y2 = std::round(std::max(0.0f, std::min(br.y * scale_y, 1.0f * PROTO_H_)));

        cv::Mat result;
        const cv::Mat &coefficient = coefficients.row(n);  // (1 x k)
        // 对每一个边界框系数执行矩阵乘法: (1 x k) * (k x hw) = (1 x hw)
        cv::gemm(coefficient, prototypes, 1.0, cv::Mat(), 0.0, result);

        // 重塑为二维矩阵
        cv::Mat mask = result.reshape(1, {PROTO_H_, PROTO_W_});
        //cv::imwrite(std::format("{}/{:03d}_1_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        // 在原型掩码上取边界框目标
        mask = mask(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
        //cv::imwrite(std::format("{}/{:03d}_2_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        //cv::resize(mask, mask, cv::Size(box.width, box.height), 0, 0, cv::INTER_LINEAR);
        //cv::imwrite(std::format("{}/{:03d}_3_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        // 缩放边界框: 裁切掉填充边界, 映射回推理原图
        box.x = std::round(std::max(0.0f, std::min(tl.x / scale_xy_, 1.0f * image_w_)));
        box.y = std::round(std::max(0.0f, std::min(tl.y / scale_xy_, 1.0f * image_h_)));
        box.width = std::round(std::max(0.0f, std::min(br.x / scale_xy_, 1.0f * image_w_))) - box.x;
        box.height = std::round(std::max(0.0f, std::min(br.y / scale_xy_, 1.0f * image_h_))) - box.y;
        cv::resize(mask, mask, cv::Size(box.width, box.height), 0, 0, cv::INTER_LINEAR);
        //cv::imwrite(std::format("{}/{:03d}_3_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        results_[n].mask = mask > mask_threshold;
    }
}

// 主要优化矩阵乘法函数实现
// coefficients: 掩码系数矩阵 (n x k)
// prototypes: 原型掩码矩阵 (k x (h*w))
void SegmentContext::MatrixMultiplyOptimize(const cv::Mat &coefficients, const cv::Mat &prototypes, const float mask_threshold) {
    // 输入数据有效性检查
    if (coefficients.empty() || prototypes.empty()) {
        throw std::invalid_argument("输入矩阵不能为空");
    }
    // 掩码系数维度
    if (prototypes.rows != coefficients.cols) {
        throw std::invalid_argument("系数维度与原型维度不匹配");
    }
    // 原型掩码空间维度
    if (prototypes.cols != PROTO_H_ * PROTO_W_) {
        throw std::invalid_argument("原型掩码空间维度与指定高度宽度不匹配");
    }

    // 根据检出目标数量选择计算策略
    if (coefficients.rows <= chunk_size_) {
        // 单线程处理小批次或禁用多线程情况
        MatrixMultiplyChunk(coefficients, prototypes, cv::Range(0, coefficients.rows), mask_threshold);
        return;
    }

    // 多线程并行处理大批次数据 动态负载均衡分配
    const int32_t chunks = (coefficients.rows + chunk_size_ - 1) / chunk_size_;
    const int32_t threads_needed = std::min(num_threads_, chunks);
    const int32_t chunk_per_thread = chunks / threads_needed;
    const int32_t chunk_remainders = chunks % threads_needed;

    // 创建异步任务容器
    std::vector<std::future<void>> futures;
    futures.reserve(threads_needed);

    // 分配计算任务给各线程
    int32_t start_idx = 0;
    for (int32_t t = 0; t < threads_needed; ++t) {
        // 计算当前线程负责的分块数
        const int32_t cur_num = chunk_per_thread + (t < chunk_remainders ? 1 : 0);
        const int32_t end_idx = std::min(start_idx + cur_num * chunk_size_, coefficients.rows);

        // 创建异步计算任务
        futures.emplace_back(
            std::async(std::launch::async, [this, &coefficients, &prototypes, start_idx, end_idx, mask_threshold]() {
                this->MatrixMultiplyChunk(coefficients, prototypes, cv::Range(start_idx, end_idx), mask_threshold);
            })
        );

        // 更新下一线程的起始位置
        start_idx = end_idx;
    }

    // 等待所有并行任务完成
    for (auto &future : futures) {
        future.wait();
    }
}