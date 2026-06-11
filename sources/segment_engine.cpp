#include "segment_engine.h"
#include "nvinfer_utils.h"

#include "NvOnnxParser.h"
#include "NvInferPlugin.h"

#include <fstream>
#include <filesystem>
#include <algorithm>


SegmentEngine::SegmentEngine() {
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

SegmentEngine::~SegmentEngine() {
    StopWorkers();
    contexts_.clear();
    context_.destroy_context();
    delete engine_;
    delete runtime_;
}

bool SegmentEngine::get_engine(const std::string &model_file) {
    std::filesystem::path file_path(model_file);
    const std::string extension = file_path.extension().string();
    if (extension == ".onnx") {
        file_path.replace_extension("engine");
        if (std::filesystem::exists(file_path)) {
            return load_network_engine(file_path.string());
        }
        return load_network_onnx(model_file);
    }
    if (extension == ".engine") {
        return load_network_engine(model_file);
    }
    throw std::runtime_error("Unknown model extension: " + extension);
}

bool SegmentEngine::load_network_onnx(const std::string &model_file) {
    const auto stage1 = std::chrono::system_clock::now();
    constexpr auto parse_verbose = static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO);
    // 显式批处理模式允许开发者明确指定输入张量的批处理维度(通常为第0维), 并支持更灵活的动态形状配置, 而隐式批处理模式则由TensorRT自动管理批处理维度.
    // 这意味着无论实际推理时的批次大小如何变化, 模型都会按照这个固定的批次大小进行推理. (Ignored: always "explicit batch" in TensorRT 10.0)
    //constexpr auto creation_flags = (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    constexpr auto creation_flags = 0;

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

bool SegmentEngine::load_network_engine(const std::string &engine_file) {
    const auto stage1 = std::chrono::system_clock::now();
    std::ifstream in_file(engine_file, std::ios::in | std::ios::binary | std::ios::ate);
    if (!in_file.good()) {
        SPDLOG_INFO("TensorRT open engine model not success: {}", engine_file);
        return false;
    }

    SPDLOG_INFO("TensorRT version: {}", NV_TENSORRT_VERSION);
    const std::streamsize model_size = in_file.tellg();
    in_file.seekg(0, std::ios::beg);
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

void SegmentEngine::get_model_dimensions() {
    const auto stage1 = std::chrono::system_clock::now();
    const auto nbIoTensors = engine_->getNbIOTensors();
    if (nbIoTensors != 3) {
        SPDLOG_ERROR("TensorRT unexpect NbIOTensors: {}", nbIoTensors);
        throw std::runtime_error(std::format("TensorRT unexpect NbIOTensors: {}", nbIoTensors));
    }

    auto const tensorName1 = engine_->getIOTensorName(input_index_);
    if (tensorName1 != std::string(INPUT_BLOB_NAME)) {
        SPDLOG_ERROR("TensorRT unexpect input IOTensorName: {}", tensorName1);
        throw std::runtime_error(std::format("TensorRT unexpect input IOTensorName: {}", tensorName1));
    }
    auto const tensorName2 = engine_->getIOTensorName(probe_index_);
    if (tensorName2 != std::string(OUTPUT1_BLOB_NAME)) {
        SPDLOG_ERROR("TensorRT unexpect output1 IOTensorName: {}", tensorName2);
        throw std::runtime_error(std::format("TensorRT unexpect output1 IOTensorName: {}", tensorName2));
    }
    auto const tensorName3 = engine_->getIOTensorName(proto_index_);
    if (tensorName3 != std::string(OUTPUT2_BLOB_NAME)) {
        SPDLOG_ERROR("TensorRT unexpect output2 IOTensorName: {}", tensorName3);
        throw std::runtime_error(std::format("TensorRT unexpect output2 IOTensorName: {}", tensorName3));
    }

    //由于模型的推理是在GPU上进行的, 所以会存在搬运输入、输出数据的操作, 因此有必要在GPU上创建内存区域用于存放输入、输出数据. 模型输入、输出的尺寸可以通过ICudaEngine对象的接口来获取, 根据这些信息我们可以先为模型分配输入、输出缓存区.
    // 获取模型输入尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 3, 1280, 1920, 0, 0, 0, 0}})
    const nvinfer1::Dims input_dims = engine_->getTensorShape(INPUT_BLOB_NAME);
    SPDLOG_INFO("TensorRT input Dimensions: {}", input_dims);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=3, d={1, 300, 38, 0, 0, 0, 0, 0}})
    const nvinfer1::Dims output1_dims = engine_->getTensorShape(OUTPUT1_BLOB_NAME);
    SPDLOG_INFO("TensorRT output1 Dimensions: {}", output1_dims);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 32, 320, 480, 0, 0, 0, 0}})
    const nvinfer1::Dims output2_dims = engine_->getTensorShape(OUTPUT2_BLOB_NAME);
    SPDLOG_INFO("TensorRT output2 Dimensions: {}", output2_dims);

    context_.create_context(engine_);
    const auto stage2 = std::chrono::system_clock::now();
    SPDLOG_INFO("TensorRT Prepare data success, usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(stage2 - stage1).count());
}

void SegmentEngine::inspect_engine(nvinfer1::INetworkDefinition *network) const {
    //// 检查层精度.
    //std::ofstream layers1("layers1.log", std::ios::out | std::ios::binary);
    //layers1 << "TensorRT Precision of Layers:" << std::endl;
    //for (int32_t i = 0; i < network->getNbLayers(); ++i) {
    //    const auto layer = network->getLayer(i);
    //    layers1 << "Layer: " << layer->getName() << ", Precision: " << getDataType(layer->getPrecision()) << std::endl;
    //}
    //layers1.close();

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

void SegmentEngine::InitWorkers(int32_t threadNum) {
    // 分配GPU内存, 绑定数据缓冲区.
    SPDLOG_INFO("TensorRT init workers: {}", threadNum);
    workers_.reserve(threadNum);
    contexts_.reserve(threadNum);
    for (int32_t i = 0; i < threadNum; ++i) {
        contexts_.push_back({});
        auto &ctx = contexts_.back();
        ctx.create_context(engine_);
        // 添加处理线程, Lambda是在工作线程中异步执行的, 这里不能使用contexts_.back()传参.
        workers_.emplace_back([this, i]{ this->Run(contexts_[i]); });
    }
}

void SegmentEngine::StopWorkers() {
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

DetectResults SegmentEngine::RunSync(const cv::Mat &image) {
    return context_.RunSync(image);
}

std::future<DetectResults> SegmentEngine::RunAsync(const cv::Mat &image) {
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

void SegmentEngine::Run(SegmentContext &ctx) {
    SPDLOG_INFO("TensorRT Enter RunAsync, Context: {}", static_cast<void *>(&ctx));
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

        task.results_.set_value(ctx.RunSync(task.image_));
    }
    SPDLOG_INFO("TensorRT Leave RunAsync, Context: {}", static_cast<void *>(&ctx));
}