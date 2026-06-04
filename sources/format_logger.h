#ifndef __INC_FORMAT_LOGGER_H
#define __INC_FORMAT_LOGGER_H

#include "spdlog/spdlog.h"
#include "opencv2/opencv.hpp"

namespace std {
template<>
struct formatter<cv::Mat> : std::formatter<std::string> {
    auto format(const cv::Mat &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        const std::string str = std::format("[{}x{},{}]", val.cols, val.rows, reinterpret_cast<const void *>(&val));
        return std::formatter<std::string>::format(str, ctx);
    }
};

template<typename T>
struct formatter<cv::Rect_<T>> : std::formatter<std::string> {
    auto format(cv::Rect_<T> const &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream ss;
        ss << "[" << val.x << ", " << val.y << ", " << val.width << ", " << val.height << "]";
        return std::formatter<std::string>::format(ss.str(), ctx);
    }
};

template<>
struct formatter<cv::Range> : std::formatter<std::string> {
    formatter<int> formatter_;
    template <typename FormatContext>
    auto format(cv::Range const &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream ss;
        ss << "[" << val.start << ", " << val.end << "]";
        return std::formatter<std::string>::format(ss.str(), ctx);
    }
};

template<typename T>
struct formatter<cv::Point_<T>> : std::formatter<std::string> {
    auto format(cv::Point_<T> const &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream ss;
        ss << "[" << val.x << ", " << val.y << "]";
        return std::formatter<std::string>::format(ss.str(), ctx);
    }
};

template<typename T>
struct formatter<cv::Size_<T>> : std::formatter<std::string> {
    auto format(const cv::Size_<T> val, std::format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream ss;
        ss << "[" << val.height << ", " << val.width << "]";
        return std::formatter<std::string>::format(ss.str(), ctx);
    }
};

template<>
struct formatter<cv::Size2i> : std::formatter<std::string> {
    auto format(const cv::Size2i &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        const auto str = std::format("[{}, {}]", val.height, val.width);
        return std::formatter<std::string>::format(str, ctx);
    }
};

template<>
struct formatter<cv::MatSize> : std::formatter<std::string> {
    auto format(const cv::MatSize &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream ss;
        const cv::Size size = val();
        ss << "[" << size.height << ", " << size.width << "]";
        return std::formatter<std::string>::format(ss.str(), ctx);
    }
};

template<typename T>
struct formatter<std::vector<T>> : std::formatter<std::string> {
    auto format(const std::vector<T> &val, std::format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < val.size(); ++i) {
            if (i != 0) { ss << ", "; }
            ss << val[i];
        }
        ss << "]";
        return std::formatter<std::string>::format(ss.str(), ctx);
    }
};
} //namespace std
#endif //__INC_FORMAT_LOGGER_H