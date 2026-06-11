#ifndef __INC_SEGMENT_UTILS_H
#define __INC_SEGMENT_UTILS_H

#include "opencv2/opencv.hpp"
#include <ranges>


struct DetectResult {
    float           score;      //结果置信度
    int32_t         label;      //结果类型Id
    cv::Rect2f      bbox;       //目标边界框
    cv::Mat         mask;       //目标掩码
};
using DetectResults = std::vector<DetectResult>;


void DrawPred(cv::Mat &image, const std::vector<DetectResult> &results);


inline std::string trim(const std::string &s) {
    if (s.empty()) {
        return s;
    }

    // 常见空白字符(C风格isspace范围)
    static constexpr std::string WHITESPACE = " \t\n\r\f\v";
    const size_t s_idx = s.find_first_not_of(WHITESPACE);
    if (s_idx == std::string::npos) {    // 全空白
        return "";
    }

    const size_t e_idx = s.find_last_not_of(WHITESPACE);
    return s.substr(s_idx, (e_idx - s_idx) + 1);
}
#endif //__INC_SEGMENT_UTILS_H