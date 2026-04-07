#ifndef __INC_NVIDIA_LOGGER_H
#define __INC_NVIDIA_LOGGER_H

#include <map>
#include <string>

#include "spdlog/spdlog.h"
#include "NvInferRuntime.h"

namespace std {
template<>
struct formatter<nvinfer1::Dims> : std::formatter<std::string> {
    auto format(const nvinfer1::Dims &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < val.nbDims; ++i) {
            if (i != 0) { ss << ", "; }
            ss << val.d[i];
        }
        ss << "]";
        return std::formatter<std::string>::format(ss.str(), ctx);
    }
};

//template<>
//struct formatter<nvinfer1::Dims> {
//    template <typename FormatContext>
//    auto format(const nvinfer1::Dims &val, FormatContext &ctx) const {
//        std::stringstream ss;
//        ss << "[";
//        for (size_t i = 0; i < val.nbDims; ++i) {
//            if (i != 0) { ss << ", "; }
//            ss << val.d[i];
//        }
//        ss << "]";
//        return std::format_to(ctx.out(), "{}", ss.str());
//    }
//
//    constexpr auto parse(std::format_parse_context &ctx) -> decltype(ctx.begin()) {
//        return ctx.begin();
//    }
//};
} //namespace std

class NvLogger : public nvinfer1::ILogger {
public:
    static NvLogger &GetInstance();

    void log(Severity severity, const nvinfer1::AsciiChar *msg) noexcept override;
    void SetSeverity(Severity severity);

private:
    static std::string GetSeverity(Severity severity);
    Severity severity_{Severity::kINFO};
};

inline std::string getDataType(const nvinfer1::DataType &v) {
    static const std::map<nvinfer1::DataType, std::string> vDataType {
            {nvinfer1::DataType::kFLOAT, "DataType.FLOAT"},
            {nvinfer1::DataType::kHALF, "DataType.HALF"},
            {nvinfer1::DataType::kINT8, "DataType.INT8"},
            {nvinfer1::DataType::kINT32, "DataType.INT32"},
            {nvinfer1::DataType::kBOOL, "DataType.BOOL"},
            {nvinfer1::DataType::kUINT8, "DataType.UINT8"},
            {nvinfer1::DataType::kFP8, "DataType.FP8"},
            {nvinfer1::DataType::kBF16, "DataType.BF16"},
            {nvinfer1::DataType::kINT64, "DataType.INT64"},
            {nvinfer1::DataType::kINT4, "DataType.INT4"},
            {nvinfer1::DataType::kFP4, "DataType.FP4"},
        };
    const auto it = vDataType.find(v);
    return it != vDataType.end() ? it->second : "";
}
#endif //__INC_NVIDIA_LOGGER_H