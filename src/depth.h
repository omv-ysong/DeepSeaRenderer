#ifndef DEPTH_H
#define DEPTH_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace uwc
{

class DepthMap
{
public:
    DepthMap() = default;
    DepthMap(const cv::Mat &);

    cv::Mat &GetNormal(const Eigen::Matrix3d &k_inverse);
    cv::Mat &GetNormal(const Eigen::Matrix3d &k_inverse, std::string write_path);

    double mean()
    {
        return mean_depth_;
    }


private:
    cv::Mat m_depth_;
    cv::Mat m_normal_;
    double mean_depth_ = 0.0;
    Eigen::Matrix3d k_mat_;

    void EstimateNormal(const Eigen::Matrix3d &k_inverse);
    void Calculate3DPoints(const cv::Mat &image, double local);

};

}


#endif // DEPTH_H
