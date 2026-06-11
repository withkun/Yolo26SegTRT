#ifndef __INC_NVINFER_UTILS_H
#define __INC_NVINFER_UTILS_H

#include <map>
#include <string>
#include <ranges>

#include "format_logger.h"
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
} //namespace std

inline bool operator<(const nvinfer1::Dims &lsh, const nvinfer1::Dims &rsh) {
    if (lsh.nbDims < rsh.nbDims) {
        return true;
    }
    for (int32_t i = 0; i < lsh.nbDims; i++) {
        if (lsh.d[i] < rsh.d[i]) {
            return true;
        }
    }
    return false;
}

struct DimsCompare {
    bool operator()(const nvinfer1::Dims &lsh, const nvinfer1::Dims &rsh) const {
        if (lsh.nbDims < rsh.nbDims) {
            return true;
        }
        for (int32_t i = 0; i < lsh.nbDims; i++) {
            if (lsh.d[i] < rsh.d[i]) {
                return true;
            }
        }
        return false;
    }
};

inline int32_t DimsInBytes(const nvinfer1::Dims &v) {
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
    return it != vDataType.end() ? it->second : "unknown";
}

#endif //__INC_NVINFER_UTILS_H