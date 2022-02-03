#include "depth.h"

uwc::DepthMap::DepthMap(const cv::Mat &depth)
: m_depth_{depth}, m_normal_{depth.size(), CV_64FC3}
{
    if (m_depth_.type() != CV_32FC1)
    {
        m_depth_.convertTo(m_depth_, CV_32FC1);
    }
    cv::Scalar mean_depth_container = cv::mean(m_depth_);
    mean_depth_ = (double)(mean_depth_container.val[0]);
}

cv::Mat& uwc::DepthMap::GetNormal(const Eigen::Matrix3d &k_inverse, const unsigned int sommoth_window_size)
{
    EstimateNormal(k_inverse, sommoth_window_size);
    return m_normal_;
}

cv::Mat& uwc::DepthMap::GetNormal(const Eigen::Matrix3d &k_inverse, const unsigned int sommoth_window_size, std::string write_path)
{
    EstimateNormal(k_inverse, sommoth_window_size);

    cv::Mat writer;
    m_normal_.convertTo(writer, CV_32FC3);
    imwrite(write_path, writer);

    return m_normal_;
}

void uwc::DepthMap::EstimateNormal(const Eigen::Matrix3d &k_inverse, const unsigned int sommoth_window_size)
{
    cv::Vec3d normal_init(0.0, 0.0, -1.0);

#pragma omp parallel for
    for (int r = 0; r < m_depth_.rows; r++)
    {
        for (int c = 0; c < m_depth_.cols; c++)
        {
          // neighbour pixel index
          //   d4
          //d2    d1   ->c
          //   d3

          if(r==0 || r==(m_depth_.rows-1) || c==0 || c==(m_depth_.cols-1))
          {
              m_normal_.at<cv::Vec3d>(r, c) = normal_init;
          }
          else{
              float depth_val = m_depth_.at<float>(r, c);
              if( depth_val < 0.01 || depth_val > 20.0){
                  m_normal_.at<cv::Vec3d>(r, c) = normal_init;
              }
              else{
                  double d1 = (double)m_depth_.at<float>(r, c + 1);
                  double d2 = (double)m_depth_.at<float>(r, c - 1);
                  double d3 = (double)m_depth_.at<float>(r + 1, c);
                  double d4 = (double)m_depth_.at<float>(r - 1, c);

                  Eigen::Vector3d p_1_homo((double)c+1.0, (double)r, 1.0);
                  Eigen::Vector3d p_2_homo((double)c-1.0, (double)r, 1.0);
                  Eigen::Vector3d p_3_homo((double)c, (double)r+1.0, 1.0);
                  Eigen::Vector3d p_4_homo((double)c, (double)r-1.0, 1.0);

                  Eigen::Vector3d v1, v2, v3, v4;
                  v1[0] = d1*(k_inverse(0,0)*p_1_homo[0] + k_inverse(0,2));
                  v1[1] = d1*(k_inverse(1,1)*p_1_homo[1] + k_inverse(1,2));
                  v1[2] = d1;
                  v2[0] = d2*(k_inverse(0,0)*p_2_homo[0] + k_inverse(0,2));
                  v2[1] = d2*(k_inverse(1,1)*p_2_homo[1] + k_inverse(1,2));
                  v2[2] = d2;
                  v3[0] = d3*(k_inverse(0,0)*p_3_homo[0] + k_inverse(0,2));
                  v3[1] = d3*(k_inverse(1,1)*p_3_homo[1] + k_inverse(1,2));
                  v3[2] = d3;
                  v4[0] = d4*(k_inverse(0,0)*p_4_homo[0] + k_inverse(0,2));
                  v4[1] = d4*(k_inverse(1,1)*p_4_homo[1] + k_inverse(1,2));
                  v4[2] = d4;

                  Eigen::Vector3d vP4P3 = v3-v4; vP4P3.normalize();
                  Eigen::Vector3d vP2P1 = v1-v2; vP2P1.normalize();

                  Eigen::Vector3d normal = vP4P3.cross(vP2P1);
                  normal.normalize();
                  m_normal_.at<cv::Vec3d>(r, c) = cv::Vec3d(normal[0], normal[1], normal[2]);
              }
          }
         }
     }

    if(sommoth_window_size == 3 || sommoth_window_size == 5){ // extra smoothing
      cv::Mat m_normal_XYZ[3];
      cv::split(m_normal_, m_normal_XYZ);
      cv::Mat grad_x, grad_y;
      cv::Sobel(m_normal_XYZ[1], grad_x, -1, 1, 0, 3);
      cv::Sobel(m_normal_XYZ[1], grad_y, -1, 0, 1, 3);
      cv::Mat normal_median, m_normal_32f;
      m_normal_.convertTo(m_normal_32f, CV_32FC3);
      cv::medianBlur(m_normal_32f, normal_median, sommoth_window_size);
      cv::medianBlur(normal_median, normal_median, sommoth_window_size);

      for(int r=0; r<m_normal_.rows; r++)
        {
            for(int c=0; c<m_normal_.cols; c++)
            {
                cv::Vec3d normal_vec = m_normal_.at<cv::Vec3d>(r,c);

                // these pixels' normals will be replaced by smoothed median values
                if(abs(normal_vec(2))<0.2)
                {
                    if(abs(grad_x.at<double>(r,c)) > 0.4 || abs(grad_y.at<double>(r,c)) > 0.4){
                        cv::Vec3f normal_median_val = normal_median.at<cv::Vec3f>(r,c);
                        Eigen::Vector3d normal_median_val_eigen((double)normal_median_val(0), (double)normal_median_val(1), (double)normal_median_val(2));
                        normal_median_val_eigen.normalize();
                        m_normal_.at<cv::Vec3d>(r,c) = cv::Vec3d(normal_median_val_eigen[0], normal_median_val_eigen[1], normal_median_val_eigen[2]);
                    }
                }
                else
                    continue;
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
