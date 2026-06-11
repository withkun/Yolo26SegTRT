#include "media_reader.h"
#include "segment_utils.h"


// 递归展开通配符路径, 返回所有匹配文件
std::vector<std::string> expandGlob(const std::string &pattern) {
    std::vector<std::string> all_files;
    {
        std::string file_item;
        std::stringstream ss(pattern);
        while (std::getline(ss, file_item, ';')) {
            file_item = trim(file_item);
            if (file_item.empty()) continue;
            std::vector<std::string> im_files;
            cv::glob(file_item, im_files, false);
            all_files.insert(all_files.end(), im_files.begin(), im_files.end());
        }
    }
    return all_files;
}

MediaReader::MediaReader(const std::string &path) : source_(path) {
    all_files_ = expandGlob(source_);
    if (all_files_.size() == 1) {
        capture_.open(all_files_.front());
    } else if (!all_files_.empty()) {
        image_ = cv::imread(all_files_.front(), cv::IMREAD_UNCHANGED);
    }
}

MediaReader::~MediaReader() {
    if (all_files_.size() == 1) {
        capture_.release();
    }
}

// 视频帧生成器
std::generator<FrameData> MediaReader::read_video_file() {
    std::filesystem::path path(all_files_.front());
    const std::string name = path.stem().string();

    cv::Mat frame;
    int32_t index = 0;
    while (capture_.isOpened() && capture_.read(frame)) {
        const std::string frame_name = std::format("{}_{:04d}.png", name, index);
        // 协程: yield当前帧, 克隆数据以确保生命周期安全
        co_yield FrameData(frame.clone(), frame_name, index++);
    }
}

std::generator<FrameData> MediaReader::read_image_list() {
    for (const auto &&[index, filename] : all_files_ | std::views::enumerate) {
        cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        // 协程: yield当前帧, 克隆数据以确保生命周期安全
        co_yield FrameData(std::move(frame), filename, index);
    }
}

// 核心接口：返回生成器, 外部直接遍历
std::generator<FrameData> MediaReader::frames() {
    if (all_files_.size() > 1) {
        // 协程: 委托给目录读取逻辑.
        co_yield std::ranges::elements_of(read_image_list());
    } else {
        // 协程: 委托给视频读取逻辑.
        co_yield std::ranges::elements_of(read_video_file());
    }
}

bool MediaReader::empty() const {
    return count() == 0;
}

size_t MediaReader::count() const {
    if (all_files_.size() > 1) {
        return all_files_.size();
    }
    return capture_.isOpened() ? capture_.get(cv::CAP_PROP_FRAME_COUNT) : 0;
}

int32_t MediaReader::width() const {
    if (all_files_.size() > 1) {
        return image_.empty() ? 0 : image_.cols;
    }
    return capture_.isOpened() ? capture_.get(cv::CAP_PROP_FRAME_WIDTH) : 0;
}

int32_t MediaReader::height() const {
    if (all_files_.size() > 1) {
        return image_.empty() ? 0 : image_.rows;
    }
    return capture_.isOpened() ? capture_.get(cv::CAP_PROP_FRAME_HEIGHT) : 0;
}