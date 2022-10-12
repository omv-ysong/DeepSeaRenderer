#ifndef DEPTH_H
#define DEPTH_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace uwc
{

class DepthMap
{
public:
    DepthMap() = default;
    DepthMap(const cv::Mat &);

    // compute normal map from depth
    cv::Mat &GetNormal(const Eigen::Matrix3d &k_inverse, const unsigned int sommoth_window_size);
    cv::Mat &GetNormal(const Eigen::Matrix3d &k_inverse, const unsigned int sommoth_window_size, std::string write_path);

    double mean()
    {
        return mean_depth_;
    }


private:
    cv::Mat m_depth_;
    cv::Mat m_normal_;
    double mean_depth_ = 0.0;
    Eigen::Matrix3d k_mat_;
    unsigned int sommoth_window_size_ = 0;

    void EstimateNormal(const Eigen::Matrix3d &k_inverse, const unsigned int sommoth_window_size);

};

}


#endif // DEPTH_H
