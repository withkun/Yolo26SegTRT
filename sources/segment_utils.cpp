#include "segment_utils.h"


std::vector<cv::Scalar> label_colormap() {
    std::vector<cv::Scalar> colormap(256);
    for (int i = 0; i < 256; ++i) {
        // 提取标签i的8个二进制位
        const uint8_t b0 = (i >> 0) & 1;
        const uint8_t b1 = (i >> 1) & 1;
        const uint8_t b2 = (i >> 2) & 1;
        const uint8_t b3 = (i >> 3) & 1;
        const uint8_t b4 = (i >> 4) & 1;
        const uint8_t b5 = (i >> 5) & 1;
        const uint8_t b6 = (i >> 6) & 1;
        const uint8_t b7 = (i >> 7) & 1;
        // 合成RGB通道色彩值.
        const uint8_t r = (b0 << 7) | (b3 << 6) | (b6 << 5);
        const uint8_t g = (b1 << 7) | (b4 << 6) | (b7 << 5);
        const uint8_t b = (b2 << 7) | (b5 << 6);
        colormap[i] = cv::Scalar(b, g, r);
    }
    return colormap;
}

void DrawPred(cv::Mat &image, const std::vector<DetectResult> &results) {
    const double fontScale = 0.5;
    const int32_t thickness = 1;
    const int32_t fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const int32_t lineType = cv::LINE_8;

    // 转三通道绘图
    const static std::vector<cv::Scalar> LABEL_COLORMAP = label_colormap();
    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }

    cv::Mat all_mask = image.clone();
    for (auto idx = 0; idx < results.size(); ++idx) {
        const auto &[id, confidence, box, mask] = results[idx];
        const auto color = LABEL_COLORMAP[idx % LABEL_COLORMAP.size()];
        cv::rectangle(image, box, color, thickness, lineType);

        // mask是一个与矩阵box大小相同的单通道二值掩码
        int32_t x1 = static_cast<int32_t>(box.x);
        int32_t y1 = static_cast<int32_t>(box.y);
        if (!mask.empty()) {
            all_mask(box).setTo(color, mask);
        }

        int32_t baseLine;
        std::string text = std::format("{}:{:.3f}", id, confidence);
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine);
        y1 = std::max(y1, textSize.height);
        cv::putText(image, text, cv::Point(x1, y1), fontFace, fontScale, color, thickness, lineType);
    }

    cv::addWeighted(image, 0.7, all_mask, 0.3, 0, image); //将mask加在原图上面
}