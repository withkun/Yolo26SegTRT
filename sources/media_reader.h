#ifndef __INC_MEDIA_READER_H
#define __INC_MEDIA_READER_H

#include <generator>
#include <filesystem>
#include "opencv2/opencv.hpp"


// 递归展开通配符路径，返回所有匹配文件
std::vector<std::string> expandGlob(const std::string &pattern);

// 统一的数据包结构
struct FrameData {
    cv::Mat     image;      // 图像数据 (BGR格式)
    size_t      index;      // 帧索引或图片序号
    std::string source;    // 来源路径 (视频文件名或图片文件名)

    // 构造函数方便初始化
    FrameData(cv::Mat img, std::string src, size_t idx)
        : image(std::move(img)), index(idx), source(std::move(src)) {}
};

class MediaReader {
public:
    explicit MediaReader(const std::string &path);
    ~MediaReader();

    [[nodiscard]] bool empty() const;
    [[nodiscard]] size_t count() const;
    [[nodiscard]] int32_t width() const;
    [[nodiscard]] int32_t height() const;

    std::generator<FrameData> frames();

private:
    std::string                 source_;
    std::vector<std::string>    all_files_;
    cv::VideoCapture            capture_;
    cv::Mat                     image_;

    std::generator<FrameData> read_video_file();
    std::generator<FrameData> read_image_list();
};
#endif //__INC_MEDIA_READER_H