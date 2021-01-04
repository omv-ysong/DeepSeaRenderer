#ifndef RENDERING_H
#define RENDERING_H

#include <random>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "yaml-cpp/yaml.h"
#include "depth.h"

namespace uw {

int factorial(int n);
double light_RID(double &angle, int &light_type);
double get_vsf_value(double &angle);
cv::Vec3d interpolate_bs(int row, int col, const double depth_val, const std::vector<double> &thickness_array, const std::vector<cv::Mat> &slabs_lookup);

class Camera
{
public:
    Camera() = default;
    Camera(const unsigned int width, const unsigned int height)
        : width_{width}, height_{height} {}
    Camera(const cv::Mat &img)
    {
        width_ = img.cols;
        height_ = img.rows;
    }
    Camera(cv::Size img_size, double fov)
    {
        width_ = img_size.width;
        height_ = img_size.height;
        double img_width_double = (double)img_size.width;
        double img_height_double = (double)img_size.height;
        focal_length_ = img_width_double/fov;
        k_mat_ << focal_length_, 0.0, img_width_double/2.0,
                0.0, focal_length_, img_height_double/2.0,
                0.0, 0.0, 1.0;
        k_mat_inv_ = k_mat_.inverse();
    }
    Camera(const cv::Mat &img, double fov)
    {
        width_ = img.cols;
        height_ = img.rows;
        double img_width_double = (double)img.cols;
        double img_height_double = (double)img.rows;
        focal_length_ = img_width_double/fov;
        k_mat_ << focal_length_, 0.0, img_width_double/2.0,
                0.0, focal_length_, img_height_double/2.0,
                0.0, 0.0, 1.0;
        k_mat_inv_ = k_mat_.inverse();
    }

    Eigen::Vector3d white_balance = Eigen::Vector3d(1.0, 1.0, 1.0);

    double f() const
    {
        return focal_length_;
    }
    int width() const
    {
        return width_;
    }
    int height() const
    {
        return height_;
    }
    cv::Size img_size() const
    {
        return cv::Size(width_, height_);
    }
    Eigen::Matrix3d K() const
    {
        return k_mat_;
    }
    Eigen::Matrix3d invK() const
    {
        return k_mat_inv_;
    }

    inline virtual void SetWB(const Eigen::Vector3d wb)
    {
        white_balance = wb;
    }
    inline virtual void SetIntrinsics(const cv::Size size, const double fov)
    {
        width_ = size.width;
        height_ = size.height;
        double img_width_double = (double)size.width;
        double img_height_double = (double)size.height;
        focal_length_ = img_width_double/fov;
        k_mat_ << focal_length_, 0.0, img_width_double/2.0,
                0.0, focal_length_, img_height_double/2.0,
                0.0, 0.0, 1.0;
        k_mat_inv_ = k_mat_.inverse();
    }

private:
    double focal_length_ = 1.0;
    Eigen::Matrix3d k_mat_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d k_mat_inv_ = Eigen::Matrix3d::Identity();
    unsigned int width_ = 0.0;
    unsigned int  height_ = 0.0;

};

class Light
{
public:
    Light() = default;
    Light(const Eigen::Vector3d position, const Eigen::Vector3d rotation_angles)
        : position_{position}, rotation_angles_{rotation_angles}
    {
        R_ = Eigen::AngleAxisd(rotation_angles[0],Eigen::Vector3d::UnitX())
                   * Eigen::AngleAxisd(rotation_angles[1],Eigen::Vector3d::UnitY())
                   * Eigen::AngleAxisd(rotation_angles[2], Eigen::Vector3d::UnitZ());
        direction_ = initial_direction_.transpose()*R_;
    }

    Eigen::Vector3d pos() const
    {
        return position_;
    }
    Eigen::Vector3d dir() const
    {
        return direction_;
    }
    Eigen::Vector3d spectrum() const
    {
        return spectrum_;
    }
    int RID_type() const
    {
        return RID_type_;
    }

    inline virtual void SetSpecturm(const Eigen::Vector3d spectrum)
    {
        spectrum_ = spectrum;
    }
    inline virtual void SetRIDType(const int rid_type)
    {
        RID_type_ = rid_type;
    }
    inline int GetRIDType()
    {
        return RID_type_;
    }
    virtual Eigen::Matrix3d RotationAngles2Matrix(const Eigen::Vector3d &rotation_angles)
    {
        R_ = Eigen::AngleAxisd(rotation_angles[0],Eigen::Vector3d::UnitX())
                   * Eigen::AngleAxisd(rotation_angles[1],Eigen::Vector3d::UnitY())
                   * Eigen::AngleAxisd(rotation_angles[2], Eigen::Vector3d::UnitZ());
        return R_;
    }

private:
    Eigen::Vector3d position_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    Eigen::Vector3d rotation_angles_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    Eigen::Vector3d initial_direction_ = Eigen::Vector3d(0.0, 0.0, 1.0);
    Eigen::Vector3d direction_ = initial_direction_;
    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d spectrum_ = Eigen::Vector3d(1.0, 1.0, 1.0);
    int RID_type_ = 0;

};

class Slab
{
public:
    Slab() = default;
    Slab(const cv::Size &img_size)
    {
        values_ = cv::Mat(img_size, CV_64FC3, cv::Scalar(0.0, 0.0, 0.0));
    }
    //Slab(const double thickness, const cv::Mat &img)
    //    : thickness_{thickness}, values_{img} {}   // this one has risks, will be removed
    Slab(const double thickness, const cv::Size &img_size)
    {
        thickness_ = thickness;
        depth_ = 0.5*thickness;
        values_ = cv::Mat(img_size, CV_64FC3, cv::Scalar(0.0, 0.0, 0.0));
    }

    inline virtual void SetThickness(const double new_thickness)
    {
        thickness_ = new_thickness;
    }
    inline virtual void SetDepth(const double new_depth)
    {
        depth_ = new_depth;
    }
    inline virtual void SetVals(const cv::Mat new_values)
    {
        values_ = new_values;
    }
    inline virtual void SetPixel(const unsigned int &row, const unsigned int &col, cv::Vec3d &new_pixel)
    {
        values_.at<cv::Vec3d>(row,col) = new_pixel;
    }

    cv::Mat values()
    {
        return values_;
    }
    double thickness()
    {
        return thickness_;
    }
    double depth()
    {
        return depth_;
    }

private:
    cv::Mat values_;
    double thickness_ = 0.0;
    double depth_ = 0.0;
};

class VolumetricField
{
public:
    VolumetricField() = default;
    VolumetricField(const int slabs_num, const cv::Size img_size)
    {
        // initialization
        slabs_num_ = slabs_num;
        Slab each_slab(img_size);
        for(int i=0; i<slabs_num; i++)
        {
            slabs_.emplace_back(each_slab);
        }
    }

    inline void Initialize(const int slabs_num, const cv::Size img_size)
    {
        slabs_num_ = slabs_num;
        Slab each_slab(img_size);
        for(int i=0; i<slabs_num; i++)
        {
            slabs_.emplace_back(each_slab);
        }
    }
    inline void SetAtteCoeff(const Eigen::Vector3d attenuation)
    {
        attenuation_coeff_ = attenuation;
    }
    inline void SetThickness(const int id, const double new_thickness)
    {
        slabs_[id].SetThickness(new_thickness);
    }
    inline void SetDepth(const int id, const double new_depth)
    {
        slabs_[id].SetDepth(new_depth);
    }
    inline void SetSlabVal(const int id, const cv::Mat &new_values)
    {
        slabs_[id].SetVals(new_values);
    }
    inline void SetSlabPixel(const int id, const unsigned int &row, const unsigned int &col, cv::Vec3d &new_pixel)
    {
        slabs_[id].SetPixel(row, col, new_pixel);
    }
    inline Eigen::Vector3d GetAtteCoeff()
    {
        return attenuation_coeff_;
    }
    inline double GetThickness(const int id)
    {
        return slabs_[id].thickness();
    }
    inline double GetDepth(const int id)
    {
        return slabs_[id].depth();
    }
    inline cv::Mat GetSlab(const int id)
    {
        return slabs_[id].values();
    }
    inline int GetSlabNum()
    {
        return slabs_num_;
    }

private:
    int slabs_num_ = 1;
    std::vector<Slab> slabs_ = {};
    Eigen::Vector3d attenuation_coeff_ = Eigen::Vector3d(0.0, 0.0, 0.0);
};

class Renderer
{
public:

    enum SlabSamplingMethod
    {
        ADAPTIVE = 0,
        EQUAL_DISTANCE = 1
    };

    Renderer() = default;
    void SetConfig(const std::string yaml_file);
    void CalculateSampling();
    void RenderFS(double average_depth, cv::Mat &slab);
    void ComputeSlabBS();
    void AccumulateBS(); // compute accumulated field
    void WriteSlabs(std::string outpute_path);
    void WriteBSField(std::string path);
    void RenderUnderwater(const cv::Mat &img_air, const cv::Mat &depth_map, const std::string &output_path);
    cv::Mat RenderUnderwater(const cv::Mat &img_air, const cv::Mat &depth_map);
    cv::Vec3d interpolate_bs(int row, int col, const double &depth_val);

    void WriteDoubleMatTo8Bit(const cv::Mat &double_mat, const std::string &write_path);
    cv::Mat ConvertDoubleMatTo8Bit(const cv::Mat &double_mat);
    void ConvertDoubleMatTo8Bit(const cv::Mat &double_mat, std::string &output_name);

private:
    int slab_sampling_method_ = 1;
    double scale_factor_ = 1.0;
    double scale_factor_bs_ = 1.0;
    bool render_back_scatter_ = false;
    double volumetric_max_depth_ = 0.0;
    int num_volumetric_slabs_ = 0;
    bool write_slab_ = false;
    Camera cam_;
    Eigen::Vector3d white_balance_;
    Eigen::Vector3d attenuation_;
    Eigen::Vector3d light_spectrum_;
    int light_type_ = 0;
    std::vector<Light> lights_ = {};
    unsigned int num_light_ = 0;
    VolumetricField vol_field_;
    bool write_uw_img_in_exr_ = false;
    bool auto_iso_ = false;
};



void GenerateBSPlot(std::string output_file, double max_dist, double sampling_dist);
void generate_flat_seefloor_dataset(int imgNum, cv::Size imgSize, double fov, std::string outputPath);
Eigen::Vector3d ray_plane_intersection(Eigen::Vector3d &rayVector, Eigen::Vector3d &rayPoint, Eigen::Vector3d &planeNormal, Eigen::Vector3d &planePoint);  // algorithm from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#C.2B.2B
void RenderPlanImage(int render_img_num, std::string outpu_path);
void ExrImageAmplity(std::string exr_filename, float times);

}  // namespace

#endif // RENDERING_H
