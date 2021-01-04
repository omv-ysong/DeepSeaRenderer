#include "depth.h"

uwc::DepthMap::DepthMap(const cv::Mat &depth)
: m_depth_{depth}
, m_normal_{depth.size(), CV_64FC3}
{
    if (m_depth_.type() != CV_32FC1)
    {
        m_depth_.convertTo(m_depth_, CV_32FC1);
    }
    cv::Scalar mean_depth_container = cv::mean(m_depth_);
    mean_depth_ = (double)(mean_depth_container.val[0]);
}

cv::Mat& uwc::DepthMap::GetNormal(const Eigen::Matrix3d &k_inverse)
{
    EstimateNormal(k_inverse);
    return m_normal_;
}

cv::Mat& uwc::DepthMap::GetNormal(const Eigen::Matrix3d &k_inverse, std::string write_path)
{
    EstimateNormal(k_inverse);

    cv::Mat writer;
    m_normal_.convertTo(writer, CV_32FC3);
    imwrite(write_path, writer);


    return m_normal_;
}

void uwc::DepthMap::EstimateNormal(const Eigen::Matrix3d &k_inverse)
{
    #pragma omp parallel for
    for (int x = 0; x < m_depth_.rows; ++x)
    {
        for (int y = 0; y < m_depth_.cols; ++y)
        {
            // use float instead of double otherwise you will not get the correct result
            // I have not figure out yet why this is happening.
            
            // neighbour pixel index
            //   d4
            //d2    d1   ->x
            //   d3


            if(x==0 || x==(m_depth_.rows-1) || y==0 || x==(m_depth_.cols-1))
            {
                cv::Vec3d normal(0.0, 0.0, -1.0);
                m_normal_.at<cv::Vec3d>(x, y) = normal;
            }
            else{
                double d1 = (double)m_depth_.at<float>(x, y + 1);
                double d2 = (double)m_depth_.at<float>(x, y - 1);
                double d3 = (double)m_depth_.at<float>(x + 1, y);
                double d4 = (double)m_depth_.at<float>(x - 1, y);

                Eigen::Vector3d p_1_homo((double)(y+1), (double)x, 1.0);
                Eigen::Vector3d p_2_homo((double)(y-1), (double)x, 1.0);
                Eigen::Vector3d p_3_homo((double)y, (double)(x+1), 1.0);
                Eigen::Vector3d p_4_homo((double)y, (double)(x-1), 1.0);

                Eigen::Vector3d v1 = k_inverse * p_1_homo;
                Eigen::Vector3d v2 = k_inverse * p_2_homo;
                Eigen::Vector3d v3 = k_inverse * p_3_homo;
                Eigen::Vector3d v4 = k_inverse * p_4_homo;

                v1.normalize();
                v2.normalize();
                v3.normalize();
                v4.normalize();

                Eigen::Vector3d P1_3d = v1 * d1/abs(v1[2]);
                Eigen::Vector3d P2_3d = v2 * d2/abs(v2[2]);
                Eigen::Vector3d P3_3d = v3 * d3/abs(v3[2]);
                Eigen::Vector3d P4_3d = v4 * d4/abs(v4[2]);

                double dzdx = (d1-d2) / (P1_3d[0]-P2_3d[0]);
                double dzdy = (d3-d4) / (P3_3d[1]-P4_3d[1]);

                //float dzdx = (m_depth.at<float>(x + 1, y) - m_depth.at<float>(x - 1, y)) / 2.0;
                //float dzdy = (m_depth.at<float>(x, y + 1) - m_depth.at<float>(x, y - 1)) / 2.0;

                cv::Vec3d direction(dzdx, dzdy, -1.0);
                cv::Vec3d normal = cv::normalize(direction);
//                Eigen::Vector3d normal(dzdx, dzdy, -1.0);
//                normal.normalize();
//                cv::Vec3d normalcv(normal[0],normal[1],normal[2]);
                m_normal_.at<cv::Vec3d>(x, y) = normal;
            }


        }
    }
}

void uwc::DepthMap::Calculate3DPoints(const cv::Mat &image, double focal)
{
    // Eigen::Matrix3d KMat_cam;
    // KMat_cam << focal, 0.0, image.cols/2.0,
    //             0.0,focal, image.rows/2.0,
    //             0.0, 0.0, 1.0;
    // Eigen::Matrix3d KMat_cam_inv = KMat_cam.inverse();

    // Eigen::Vector3d p_2d_homo((double)c, (double)r, 1.0);
    // Eigen::Vector3d vec_cam2voxel = KMat_cam_inv * p_2d_homo;
    // vec_cam2voxel.normalize();
    // Eigen::Vector3d p_3d;
    // p_3d = vec_cam2voxel * current_slab_depth/abs(vec_cam2voxel[2]); // origion is at (0,0,0), so omitted
    // double d_cam2voxel = p_3d.norm(); // origion is at (0,0,0), so omitted
}
