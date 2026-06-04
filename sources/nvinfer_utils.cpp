#include "nvinfer_utils.h"


NvLogger &NvLogger::GetInstance() {
    static NvLogger instance_;
    return instance_;
}

void NvLogger::SetSeverity(Severity severity) {
    severity_ = severity;
}

std::string NvLogger::GetSeverity(Severity severity) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return "[F]";
        case Severity::kERROR:          return "[E]";
        case Severity::kWARNING:        return "[W]";
        case Severity::kINFO:           return "[I]";
        case Severity::kVERBOSE:        return "[V]";
        default:                        return "[*]";
    }
}

void NvLogger::log(Severity severity, const nvinfer1::AsciiChar *msg) noexcept {
    // suppress info-level messages
    if (severity > severity_) {
        return;
    }

    if (severity <= Severity::kERROR) {
        SPDLOG_ERROR("{}", msg);
    } else if (severity == Severity::kWARNING) {
        SPDLOG_WARN("{}", msg);
    } else if (severity == Severity::kINFO) {
        SPDLOG_INFO("{}", msg);
    } else if (severity == Severity::kVERBOSE) {
        SPDLOG_INFO("{}", msg);
    } else {
        SPDLOG_INFO("{}", msg);
    }
}