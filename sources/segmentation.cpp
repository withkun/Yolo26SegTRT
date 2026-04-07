#include "segmentation.h"
#include "nvidia_logger.h"

#include "NvOnnxParser.h"
#include "NvInferPlugin.h"

#include <fstream>
#include <filesystem>
#include <algorithm>


namespace {
const auto INPUT_BLOB_NAME   = "images";
const auto OUTPUT1_BLOB_NAME = "output0";
const auto OUTPUT2_BLOB_NAME = "output1";

constexpr float CONF_THRESHOLD = 0.25f; // 置信度阈值
constexpr float MASK_THRESHOLD = 0.20f; // 掩码二值化阈值
}

Segmentation::Segmentation() {
    NvLogger::GetInstance().SetSeverity(nvinfer1::ILogger::Severity::kINFO);
    bool plugin = initLibNvInferPlugins(&NvLogger::GetInstance(), "");
    if (!plugin) {
        SPDLOG_ERROR("TensorRT initLibNvInferPlugins failed.");
    }

    // 得到优化后的序列化模型后, 还需要创建一个IRuntime接口的实例, 然后通过其模型反序列化接口去创建一个ICudaEngine对象:
    runtime_ = nvinfer1::createInferRuntime(NvLogger::GetInstance());
    if (runtime_ == nullptr) {
        SPDLOG_ERROR("TensorRT createInferRuntime failed.");
    }
}

Segmentation::~Segmentation() {
    SPDLOG_INFO("TensorRT destroy Segmentation");
    StopWorkers();
    for (auto &ctx : contexts_) {
        destroy_context(ctx);
    }
    contexts_.clear();
    destroy_context(context_);
    delete engine_;
    delete runtime_;
}

bool Segmentation::get_engine(const std::string &model_file, const std::vector<int64_t> &image_dims) {
    std::filesystem::path file_path(model_file);
    const std::string extension = file_path.extension().string();
    if (extension == ".onnx") {
        file_path.replace_extension("engine");
        if (std::filesystem::exists(file_path)) {
            return load_network_engine(file_path.string());
        }
        return load_network_onnx(model_file, image_dims);
    }
    if (extension == ".engine") {
        return load_network_engine(model_file);
    }
    throw std::runtime_error("Unknown model extension: " + extension);
}

bool Segmentation::load_network_onnx(const std::string &model_file, const std::vector<int64_t> &image_dims) {
    const auto stage1 = std::chrono::system_clock::now();
    constexpr auto parse_verbose = static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO);
    // 显式批处理模式允许开发者明确指定输入张量的批处理维度(通常为第0维), 并支持更灵活的动态形状配置, 而隐式批处理模式则由TensorRT自动管理批处理维度.
    constexpr auto creation_flags = (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)); // 显式批处理模式

    const auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(NvLogger::GetInstance()));
    const auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(creation_flags));
    const auto parser  = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, NvLogger::GetInstance()));
    const auto config  = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    //解析网络模型
    SPDLOG_INFO("TensorRT version: {}", NV_TENSORRT_VERSION);
    if (!parser->parseFromFile(model_file.c_str(), parse_verbose)) {
        SPDLOG_INFO("TensorRT load onnx model not success: {}", model_file);
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            NvLogger::GetInstance().log(nvinfer1::ILogger::Severity::kERROR, parser->getError(i)->desc());
        }
        return false;
    }
    const auto stage2 = std::chrono::system_clock::now();
    SPDLOG_INFO("TensorRT Completed parsing of ONNX file, usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(stage2 - stage1).count());

    //模型优化
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 31);  // 当前显卡4G显存, 配置使用2GM.
    //if (builder->platformHasFastFp16()) {
    //    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    //}

    if (!image_dims.empty()) {
        nvinfer1::Dims kDims;
        kDims.nbDims = static_cast<int32_t>(image_dims.size());;
        for (size_t i = 0; i < image_dims.size(); ++i) {
            kDims.d[i] = image_dims[i];
        }

        auto *const profile = builder->createOptimizationProfile();
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMIN, kDims);    // 最小尺寸
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kOPT, kDims);    // 最优尺寸
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMAX, kDims);    // 最大尺寸
        config->addOptimizationProfile(profile);
    }

    const auto stage3 = std::chrono::system_clock::now();
    SPDLOG_INFO("TensorRT model optimize success, usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(stage3 - stage2).count());

    //设置IBuilderConfig属性后, 就可以启动优化引擎对模型进行优化了, 这个过程需要一定的时间, 在嵌入式平台上可能会比较久一点.
    //经过优化后的序列化模型被保存到IHostMemory对象中, 可以将其保存到磁盘, 下次使用时直接加载这个经过优化的模型即可, 这样可以省去等待模型优化的过程.
    SPDLOG_INFO("TensorRT Building an engine from file {}; this may take a while...", model_file);
    const auto *serialized_model = builder->buildSerializedNetwork(*network, *config);
    engine_ = runtime_->deserializeCudaEngine(serialized_model->data(), serialized_model->size());
    const auto stage4 = std::chrono::system_clock::now();
    SPDLOG_INFO("TensorRT build CUDA engine success, usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(stage4 - stage3).count());

    // 将模型序列化到engine文件中
    std::filesystem::path engine_file(model_file);
    engine_file.replace_extension("engine");
    std::ofstream out_file(engine_file.string(), std::ios::out | std::ios::binary);
    if (out_file.good()) {
        out_file.write(static_cast<const char *>(serialized_model->data()), static_cast<std::streamsize>(serialized_model->size()));
        out_file.close();
    }
    delete serialized_model;

    get_model_dimensions();
    const auto stage5 = std::chrono::system_clock::now();
    SPDLOG_INFO("TensorRT save serialized engine success, usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(stage5 - stage4).count());
    inspect_engine(network.get());

    return true;
}

bool Segmentation::load_network_engine(const std::string &engine_file) {
    const auto stage1 = std::chrono::system_clock::now();
    std::ifstream in_file(engine_file, std::ios::in | std::ios::binary | std::ios::ate);
    if (!in_file.good()) {
        SPDLOG_INFO("TensorRT open engine model not success: {}", engine_file);
        return false;
    }

    SPDLOG_INFO("TensorRT version: {}", NV_TENSORRT_VERSION);
    const std::streamsize model_size = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    auto *serialized_model = new char[model_size];
    in_file.read(serialized_model, model_size);
    engine_ = runtime_->deserializeCudaEngine(serialized_model, model_size);
    if (!engine_) {
        SPDLOG_INFO("TensorRT deserializeCudaEngine failed: {}", engine_file);
        return false;
    }
    delete [] serialized_model;

    get_model_dimensions();
    const auto stage2 = std::chrono::system_clock::now();
    SPDLOG_INFO("TensorRT load CUDA engine success, usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(stage2 - stage1).count());
    return true;
}

void Segmentation::get_model_dimensions() {
    const auto stage1 = std::chrono::system_clock::now();
    const auto nbIoTensors = engine_->getNbIOTensors();
    if (nbIoTensors != 3) {
        SPDLOG_ERROR("TensorRT unexpect NbIOTensors: {}", nbIoTensors);
        throw std::runtime_error(std::format("TensorRT unexpect NbIOTensors: {}", nbIoTensors));
    }

    input_index_   = 0;
    output1_index_ = 1;
    output2_index_ = 2;
    auto const tensorName1 = engine_->getIOTensorName(input_index_);
    if (tensorName1 != std::string(INPUT_BLOB_NAME)) {
        SPDLOG_ERROR("TensorRT unexpect input IOTensorName: {}", tensorName1);
        throw std::runtime_error(std::format("TensorRT unexpect input IOTensorName: {}", tensorName1));
    }
    auto const tensorName2 = engine_->getIOTensorName(output1_index_);
    if (tensorName2 != std::string(OUTPUT1_BLOB_NAME)) {
        SPDLOG_ERROR("TensorRT unexpect output1 IOTensorName: {}", tensorName2);
        throw std::runtime_error(std::format("TensorRT unexpect output1 IOTensorName: {}", tensorName2));
    }
    auto const tensorName3 = engine_->getIOTensorName(output2_index_);
    if (tensorName3 != std::string(OUTPUT2_BLOB_NAME)) {
        SPDLOG_ERROR("TensorRT unexpect output2 IOTensorName: {}", tensorName3);
        throw std::runtime_error(std::format("TensorRT unexpect output2 IOTensorName: {}", tensorName3));
    }

    //由于模型的推理是在GPU上进行的, 所以会存在搬运输入、输出数据的操作, 因此有必要在GPU上创建内存区域用于存放输入、输出数据. 模型输入、输出的尺寸可以通过ICudaEngine对象的接口来获取, 根据这些信息我们可以先为模型分配输入、输出缓存区.
    // 获取模型输入尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 3, 1280, 1920, 0, 0, 0, 0}})
    input_dims_ = engine_->getTensorShape(INPUT_BLOB_NAME);
    SPDLOG_INFO("TensorRT input Dimensions: {}", input_dims_);
    INPUT_N_ = static_cast<int32_t>(input_dims_.d[0]);      // 1
    INPUT_C_ = static_cast<int32_t>(input_dims_.d[1]);      // 3
    INPUT_H_ = static_cast<int32_t>(input_dims_.d[2]);      // 1280
    INPUT_W_ = static_cast<int32_t>(input_dims_.d[3]);      // 1920
    INPUT_SIZE_ = calc_dims_size(input_dims_);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=3, d={1, 300, 38, 0, 0, 0, 0, 0}})
    output1_dims_ = engine_->getTensorShape(OUTPUT1_BLOB_NAME);
    SPDLOG_INFO("TensorRT output1 Dimensions: {}", output1_dims_);
    PROBES_N_ = static_cast<int32_t>(output1_dims_.d[0]);   // 1
    PROBES_H_ = static_cast<int32_t>(output1_dims_.d[1]);   // 300
    PROBES_W_ = static_cast<int32_t>(output1_dims_.d[2]);   // 38
    OUTPUT1_SIZE_ = calc_dims_size(output1_dims_);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 32, 320, 480, 0, 0, 0, 0}})
    output2_dims_ = engine_->getTensorShape(OUTPUT2_BLOB_NAME);
    SPDLOG_INFO("TensorRT output2 Dimensions: {}", output2_dims_);
    PROTOS_N_ = static_cast<int32_t>(output2_dims_.d[0]);   // 1
    PROTOS_C_ = static_cast<int32_t>(output2_dims_.d[1]);   // 32
    PROTOS_H_ = static_cast<int32_t>(output2_dims_.d[2]);   // 320
    PROTOS_W_ = static_cast<int32_t>(output2_dims_.d[3]);   // 480
    OUTPUT2_SIZE_ = calc_dims_size(output2_dims_);

    // 创建同步推理资源.
    create_context(context_);
    const auto stage2 = std::chrono::system_clock::now();
    SPDLOG_INFO("TensorRT Prepare data success, usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(stage2 - stage1).count());
}

void Segmentation::inspect_engine(nvinfer1::INetworkDefinition *network) const {
    // 检查层精度.
    std::ofstream layers1("layers1.log", std::ios::out | std::ios::binary);
    layers1 << "TensorRT Precision of Layers:" << std::endl;
    for (int32_t i = 0; i < network->getNbLayers(); ++i) {
        const auto layer = network->getLayer(i);
        layers1 << "Layer: " << layer->getName() << ", Precision: " << getDataType(layer->getPrecision()) << std::endl;
    }
    layers1.close();

    // 层融合情况.
    const auto *inspector = engine_->createEngineInspector();
    std::ofstream layers2("layers2.log", std::ios::out | std::ios::binary);
    layers2 << "TensorRT getLayerInformation:" << std::endl;
    for (int32_t i = 0; i < engine_->getNbLayers(); ++i){
        layers2 << inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON) << ",";
    }
    layers2.close();

    // 查看层信息.
    std::ofstream layers3("layers3.log", std::ios::out | std::ios::binary);
    layers3 << "TensorRT getEngineInformation:" << std::endl;
    layers3 << inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    layers3.close();

    delete inspector;
}

void Segmentation::create_context(Context &ctx) const {
    // 分配GPU内存, 绑定数据缓冲区.
    cudaStreamCreate(&ctx.stream_);
    ctx.context_ = engine_->createExecutionContext();
    ctx.buffers_.resize(3, nullptr);
    cudaMalloc(&ctx.buffers_[input_index_],   INPUT_SIZE_   * sizeof(float));
    cudaMalloc(&ctx.buffers_[output1_index_], OUTPUT1_SIZE_ * sizeof(float));
    cudaMalloc(&ctx.buffers_[output2_index_], OUTPUT2_SIZE_ * sizeof(float));
    ctx.context_->setTensorAddress(INPUT_BLOB_NAME,   ctx.buffers_[input_index_]);
    ctx.context_->setTensorAddress(OUTPUT1_BLOB_NAME, ctx.buffers_[output1_index_]);
    ctx.context_->setTensorAddress(OUTPUT2_BLOB_NAME, ctx.buffers_[output2_index_]);
    ctx.output1_probes_ = cv::Mat::zeros(PROBES_N_ * PROBES_H_, PROBES_W_, CV_32F);
    ctx.output2_protos_ = cv::Mat::zeros(PROTOS_N_ * PROTOS_C_, PROTOS_H_ * PROTOS_W_, CV_32F);
    SPDLOG_INFO("TensorRT Init, Context: {}:{} {}:{}:{}", static_cast<void *>(&ctx), static_cast<void *>(ctx.context_), ctx.buffers_[input_index_], ctx.buffers_[output1_index_], ctx.buffers_[output2_index_]);
}

void Segmentation::destroy_context(Context &ctx) const {
    ctx.context_->setTensorAddress(INPUT_BLOB_NAME,   nullptr);
    ctx.context_->setTensorAddress(OUTPUT1_BLOB_NAME, nullptr);
    ctx.context_->setTensorAddress(OUTPUT2_BLOB_NAME, nullptr);
    for (const auto &ptr : ctx.buffers_) {
        cudaFree(ptr);
    }
    ctx.buffers_.clear();
    delete ctx.context_;
    cudaStreamDestroy(ctx.stream_);
    std::cerr << std::endl;
}

void Segmentation::InitWorkers(int32_t threadNum) {
    // 分配GPU内存, 绑定数据缓冲区.
    SPDLOG_INFO("TensorRT init workers: {}", threadNum);
    workers_.reserve(threadNum);
    contexts_.reserve(threadNum);
    for (int32_t i = 0; i < threadNum; ++i) {
        contexts_.push_back({});
        auto &ctx = contexts_.back();
        create_context(ctx);
        // 添加处理线程, Lambda是在工作线程中异步执行的, 这里不能使用contexts_.back()传参.
        workers_.emplace_back([this, i]{ this->Run(contexts_[i]); });
    }
}

void Segmentation::StopWorkers() {
    SPDLOG_INFO("TensorRT stop workers");
    {
        std::unique_lock<std::mutex> lock(this->mutex_);
        stopped_ = true;
    }

    cv_.notify_all();
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

void Segmentation::letterbox(const cv::Mat &image, Context &ctx) const {
    ctx.image_h_ = image.rows;
    ctx.image_w_ = image.cols;
    SPDLOG_INFO("TensorRT letterbox image size: {}", image.size);

    // 图像预处理方法: 计算缩放比例, 取其中较小的一侧以保持原图的宽高比.
    ctx.scale_xy_ = std::min(1.0 * INPUT_H_ / image.rows, 1.0 * INPUT_W_ / image.cols);
    const auto scaled_h = static_cast<int32_t>(image.rows * ctx.scale_xy_);
    const auto scaled_w = static_cast<int32_t>(image.cols * ctx.scale_xy_);
    ctx.offset_y_ = INPUT_H_ - scaled_h;
    ctx.offset_x_ = INPUT_W_ - scaled_w;
    SPDLOG_INFO("TensorRT letterbox scale_xy: {} offset_x: {} offset_y: {} image: {}", ctx.scale_xy_, ctx.offset_x_, ctx.offset_y_, image.ptr<void>());

    // cv::INTER_AREA适用于缩小图像, 放大图像可能得到非预期的结果.
    // 如果需要放大图像, 应该使用cv::INTER_LINEAR或cv::INTER_CUBIC.
    cv::Mat scale;
    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, scale, cv::COLOR_GRAY2RGB);
        cv::resize(scale, scale, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
    } else {
        cv::resize(image, scale, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
    }
    SPDLOG_INFO("TensorRT letterbox scale image size: {}", scale.size);

    // src: 输入图像
    // dst: 输出图像
    // top: 上侧扩展像素数
    // bottom: 下侧扩展像素数
    // left: 左侧扩展像素数
    // right: 右侧扩展像素数
    // borderType: 图像边界扩展模式
    // value: 边缘填充值, 默认值Scalar(0,0,0)
    cv::copyMakeBorder(scale, scale, 0, ctx.offset_y_, 0, ctx.offset_x_, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    //cv::imwrite("resized.png", resized);

    // image: 输入图像, 灰度图或三通道图(一般为BGR).
    // blob: 输出4维矩阵, 符合模型输入的NCHW格式. [1, C, H, W]
    // scalefactor: 缩放因子, 图像像素值的缩放比例; 图像像素减去平均值之后, 再进行缩放, 默认值是1.
    // size: 目标尺寸, 模型输入的图片尺寸.
    // mean: 图像要减去均值, 如果需要对BGR图片的三个通道分别减去不同的值, 可以使用3个值; 如果三通道图像只有1个值, 那么三个通道都减去相同的值.
    // swapRB: OpenCV中图片通道顺序是BGR, 但是假设输入顺序是RGB, 处理时可以同步转换为RGB格式, 那么就要使swapRB=true.
    // crop: 是否裁剪, 调整尺寸时是保持比例并裁剪(非拉伸), 如果crop裁剪为true, 则调整输入图像的大小, 使调整大小后的一侧等于相应的尺寸, 另一侧等于或大于, 然后从中心进行裁剪; 如果crop裁剪为false, 则直接调整大小而不进行裁剪并保留纵横比.
    // ddepth: 输出数据类型, 通常为CV_32F或CV_8U.
    cv::dnn::blobFromImage(scale, ctx.input_image_, 1.0/255.0, cv::Size(INPUT_W_, INPUT_H_), cv::Scalar(), true, false, CV_32F);
    SPDLOG_INFO("TensorRT letterbox final image size: {} blob: {}", scale.size, ctx.input_image_.ptr<void>());
}

void Segmentation::inference(Context &ctx) const {
    SPDLOG_INFO("TensorRT inference enter");
    // 异步流拷贝输入数据
    cudaMemcpyAsync(ctx.buffers_[input_index_], ctx.input_image_.ptr<float>(0), INPUT_SIZE_ * sizeof(float), cudaMemcpyHostToDevice, ctx.stream_);

    // 异步流提交推理任务
    ctx.context_->enqueueV3(ctx.stream_);

    // 异步流拷贝输出数据
    cudaMemcpyAsync(ctx.output1_probes_.ptr<float>(0), ctx.buffers_[output1_index_], OUTPUT1_SIZE_ * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream_);
    cudaMemcpyAsync(ctx.output2_protos_.ptr<float>(0), ctx.buffers_[output2_index_], OUTPUT2_SIZE_ * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream_);

    // 流同步等待处理完成
    cudaStreamSynchronize(ctx.stream_);
    SPDLOG_INFO("TensorRT inference leave");
}

void Segmentation::postprocess(const cv::Mat &image, Context &ctx, double conf_threshold, double mask_threshold) const {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // YOLO26模型输出:
    // output0输出为预测结果张量, 其维度是(1,300,38)  NHW: (x1,y1,x2,y2, score, class_id, mask1,mask2,...,mask32)
    // 38为4+1+1+32, 4为box的[cx,cy,w,h], 1个类别置信度, 1个类别标识, 32个掩膜权重(mask coefficients), 300个预测候选结果.
    // output1输出为掩膜原型张量, 其维度是(1,32,320,480), 32对应掩膜权重. 掩膜原型(320×480)需先上采样至输入图像尺寸(1280×1920).
    // 模型在推理时, 会根据每个检测框的32个掩膜权重, 对这些掩膜原型进行加权求和, 从而生成每个目标的掩膜.
    // YOLO26预测结果数据结构示例:
    // (nvinfer1::Dims{nbDims=3, d={1, 300, 38, 0, 0, 0, 0, 0}})   // 300行, 38列
    // 38为4+1+1+32, 4为box的[x1,y1,x2,y2], 1个类别置信度, 1个类别标识, 32是掩膜权重
    // box1{ {x1,y1,x2,y2}, {score}, {label}, {mask1,mask2,...,mask32} }
    // box2{ {x1,y1,x2,y2}, {score}, {label}, {mask1,mask2,...,mask32} }
    // box3{ {x1,y1,x2,y2}, {score}, {label}, {mask1,mask2,...,mask32} }
    // box4{ {x1,y1,x2,y2}, {score}, {label}, {mask1,mask2,...,mask32} }
    // box5{ {x1,y1,x2,y2}, {score}, {label}, {mask1,mask2,...,mask32} }
    // box300
    SPDLOG_INFO(std::format("TensorRT offset_x:{} offset_y:{} scale_xy:{}", ctx.offset_x_, ctx.offset_y_, ctx.scale_xy_));
    cv::Mat coefficients;  // 掩膜权重
    coefficients.reserve(PROBES_H_);
    ctx.results_.clear();
    ctx.results_.reserve(PROBES_H_);

    // 根据置信度过滤(Proposals Tensor)
    const cv::Mat &proposals = ctx.output1_probes_;
    for (int32_t row = 0; row < PROBES_H_; ++row) {
        // 访问顺序: Mat::at(row, col)
        const float score = proposals.at<float>(row, 4);
        if (score < CONF_THRESHOLD) {
            continue;
        }
        ctx.results_.push_back({});
        DetectResult &result = ctx.results_.back();
        result.id = proposals.at<float>(row, 5);
        result.confidence = score;

        const cv::Mat probe = proposals(cv::Rect(6, row, PROTOS_C_, 1)).clone();
        coefficients.push_back(probe);   // HW (1,32)

        // 边界框格式: xyxy
        const auto x1 = proposals.at<float>(row, 0);
        const auto y1 = proposals.at<float>(row, 1);
        const auto x2 = proposals.at<float>(row, 2);
        const auto y2 = proposals.at<float>(row, 3);
        result.box = cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
    }
    SPDLOG_INFO("TensorRT probe size after conf_threshold: {}", ctx.results_.size());

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 掩膜原型处理 (Mask Prototype Tensor)
    const cv::Mat &prototypes = ctx.output2_protos_;
    // 多线程矩阵乘: coefficients @ prototypes ^ T
    MatrixMultiplyOptimize(ctx, coefficients, prototypes);
    SPDLOG_INFO("TensorRT after postprocess: {}", ctx.results_.size());
}

// 分块矩阵乘法核心实现
// coefficients: 掩码系数矩阵 (n x k)
// prototypes: 原型掩码矩阵 (k x (h*w))
void Segmentation::MatrixMultiplyChunk(Context &ctx, const cv::Mat &coefficients, const cv::Mat &prototypes,
                                       const cv::Range &range) const {
    // 计算原型掩码的缩放比例
    const float scale_x = 1.0 * PROTOS_W_ / INPUT_W_;
    const float scale_y = 1.0 * PROTOS_H_ / INPUT_H_;

    const auto t0 = cv::getTickCount();
    const auto path = std::format("{}", t0);
    //std::filesystem::create_directory(path);

    // 对每个检测目标执行矩阵乘法运算
    for (int32_t n = range.start; n < range.end && n < coefficients.rows; ++n) {
        auto &box = ctx.results_[n].box;
        const auto tl = box.tl();
        const auto br = box.br();
        // 将边界框坐标转换到原型掩码空间
        const auto x1 = std::round(std::max(0.0f, std::min(tl.x * scale_x, 1.0f * PROTOS_W_)));
        const auto y1 = std::round(std::max(0.0f, std::min(tl.y * scale_y, 1.0f * PROTOS_H_)));
        const auto x2 = std::round(std::max(0.0f, std::min(br.x * scale_x, 1.0f * PROTOS_W_)));
        const auto y2 = std::round(std::max(0.0f, std::min(br.y * scale_y, 1.0f * PROTOS_H_)));

        cv::Mat result;
        const cv::Mat &coefficient = coefficients.row(n);  // (1 x k)
        // 对每一个边界框系数执行矩阵乘法: (1 x k) * (k x hw) = (1 x hw)
        cv::gemm(coefficient, prototypes, 1.0, cv::Mat(), 0.0, result);

        // 重塑为二维矩阵
        cv::Mat mask = result.reshape(1, {PROTOS_H_, PROTOS_W_});
        //cv::imwrite(std::format("{}/{:03d}_1_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        // 在原型掩码上取边界框目标
        mask = mask(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
        //cv::imwrite(std::format("{}/{:03d}_2_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        //cv::resize(mask, mask, cv::Size(box.width, box.height), 0, 0, cv::INTER_LINEAR);
        //cv::imwrite(std::format("{}/{:03d}_3_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        // 缩放边界框: 裁切掉填充边界, 映射回推理原图
        box.x = std::round(std::max(0.0f, std::min(tl.x / ctx.scale_xy_, 1.0f * ctx.image_w_)));
        box.y = std::round(std::max(0.0f, std::min(tl.y / ctx.scale_xy_, 1.0f * ctx.image_h_)));
        box.width = std::round(std::max(0.0f, std::min(br.x / ctx.scale_xy_, 1.0f * ctx.image_w_))) - box.x;
        box.height = std::round(std::max(0.0f, std::min(br.y / ctx.scale_xy_, 1.0f * ctx.image_h_))) - box.y;
        cv::resize(mask, mask, cv::Size(box.width, box.height), 0, 0, cv::INTER_LINEAR);
        //cv::imwrite(std::format("{}/{:03d}_3_{}_{}_{}_{}.png", t0, n, x1, y1, x2, y2), mask);

        ctx.results_[n].mask = mask > MASK_THRESHOLD;
    }
}

// 主要优化矩阵乘法函数实现
// coefficients: 掩码系数矩阵 (n x k)
// prototypes: 原型掩码矩阵 (k x (h*w))
void Segmentation::MatrixMultiplyOptimize(Context &ctx, const cv::Mat &coefficients, const cv::Mat &prototypes) const {
    // 输入数据有效性检查
    if (coefficients.empty() || prototypes.empty()) {
        throw std::invalid_argument("输入矩阵不能为空");
    }
    // 掩码系数维度
    if (prototypes.rows != coefficients.cols) {
        throw std::invalid_argument("系数维度与原型维度不匹配");
    }
    // 原型掩码空间维度
    if (prototypes.cols != PROTOS_H_ * PROTOS_W_) {
        throw std::invalid_argument("原型掩码空间维度与指定高度宽度不匹配");
    }

    // 根据检出目标数量选择计算策略
    if (coefficients.rows <= chunk_size_) {
        // 单线程处理小批次或禁用多线程情况
        MatrixMultiplyChunk(ctx, coefficients, prototypes, cv::Range(0, coefficients.rows));
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
            std::async(std::launch::async, [this, &ctx, &coefficients, &prototypes, start_idx, end_idx]() {
                this->MatrixMultiplyChunk(ctx, coefficients, prototypes, cv::Range(start_idx, end_idx));
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

DetectResults Segmentation::RunSync(const cv::Mat &image) {
    // OpenCV默认BGR顺序, 这里假设输入为BGR三通道图像
    letterbox(image, context_);
    inference(context_);
    postprocess(image, context_, CONF_THRESHOLD, MASK_THRESHOLD);
    return context_.results_;
}

std::future<DetectResults> Segmentation::RunAsync(const cv::Mat &image) {
    Task task;
    task.image_ = image;
    std::future<DetectResults> f = task.results_.get_future();
    {
        std::unique_lock<std::mutex> lock(this->mutex_);
        queue_.emplace(std::move(task));
    }
    cv_.notify_one();
    return f;
}

void Segmentation::Run(Context &ctx) {
    SPDLOG_INFO("TensorRT Enter RunAsync, Context: {}:{} {}:{}:{}", static_cast<void *>(&ctx), static_cast<void *>(ctx.context_), ctx.buffers_[input_index_], ctx.buffers_[output1_index_], ctx.buffers_[output2_index_]);
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(this->mutex_);
            this->cv_.wait(lock, [this] {
                return this->stopped_ || !this->queue_.empty();
            });

            if (this->stopped_ && this->queue_.empty()) {
                break;
            }

            task = std::move(this->queue_.front());
            this->queue_.pop();
        }

        // OpenCV默认BGR顺序, 这里假设输入为BGR三通道图像
        letterbox(task.image_, ctx);
        inference(ctx);
        postprocess(task.image_, ctx, ctx.scale_xy_, ctx.offset_x_);

        // 设置处理结果
        SPDLOG_INFO("classId: {}", ctx.results_.size());
        task.results_.set_value(ctx.results_);
    }
    SPDLOG_INFO("TensorRT Leave RunAsync, Context: {}:{}", static_cast<void *>(&ctx), static_cast<void *>(ctx.context_));
}