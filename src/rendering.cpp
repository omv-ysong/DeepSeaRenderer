#include "rendering.h"

double BXRA_table[10][2] =
    {{0.0, 1.0}, {10.0, 0.96}, {20.0, 0.87}, {30.0, 0.75}, {35.0, 0.63}, 
     {40.0, 0.5}, {45.0, 0.28}, {50.0, 0.2}, {60.0, 0.12}, {90.0, 0.0}}; // LED relative intensity distribution from real lab measurement

double VSF_table[55][4] = {{0.100, 5.318e+001, 6.533e+002, 3.262e+003},
                           {0.126, 4.042e+001, 4.577e+002, 2.397e+003},
                           {0.158, 3.073e+001, 3.206e+002, 1.757e+003},
                           {0.200, 2.374e+001, 2.252e+002, 1.275e+003},
                           {0.251, 1.814e+001, 1.579e+002, 9.260e+002},
                           {0.316, 1.360e+001, 1.104e+002, 6.764e+002},
                           {0.398, 9.954e+000, 7.731e+001, 5.027e+002},
                           {0.501, 7.179e+000, 5.371e+001, 3.705e+002},
                           {0.631, 5.110e+000, 3.675e+001, 2.676e+002},
                           {0.794, 3.591e+000, 2.481e+001, 1.897e+002},
                           {1.000, 2.498e+000, 1.662e+001, 1.329e+002},
                           {1.259, 1.719e+000, 1.106e+001, 9.191e+001},
                           {1.585, 1.171e+000, 7.306e+000, 6.280e+001},
                           {1.995, 7.758e-001, 4.751e+000, 4.171e+001},
                           {2.512, 5.087e-001, 3.067e+000, 2.737e+001},
                           {3.162, 3.340e-001, 1.977e+000, 1.793e+001},
                           {3.981, 2.196e-001, 1.273e+000, 1.172e+001},
                           {5.012, 1.446e-001, 8.183e-001, 7.655e+000},
                           {6.310, 9.522e-002, 5.285e-001, 5.039e+000},
                           {7.943, 6.282e-002, 3.402e-001, 3.302e+000},
                           {10.000, 4.162e-002, 2.155e-001, 2.111e+000},
                           {15.000, 2.038e-002, 9.283e-002, 9.041e-001},
                           {20.000, 1.099e-002, 4.427e-002, 4.452e-001},
                           {25.000, 6.166e-003, 2.390e-002, 2.734e-001},
                           {30.000, 3.888e-003, 1.445e-002, 1.613e-001},
                           {35.000, 2.680e-003, 9.063e-003, 1.109e-001},
                           {40.000, 1.899e-003, 6.014e-003, 7.913e-002},
                           {45.000, 1.372e-003, 4.144e-003, 5.858e-002},
                           {50.000, 1.020e-003, 2.993e-003, 4.388e-002},
                           {55.000, 7.683e-004, 2.253e-003, 3.288e-002},
                           {60.000, 6.028e-004, 1.737e-003, 2.548e-002},
                           {65.000, 4.883e-004, 1.369e-003, 2.041e-002},
                           {70.000, 4.069e-004, 1.094e-003, 1.655e-002},
                           {75.000, 3.457e-004, 8.782e-004, 1.345e-002},
                           {80.000, 3.019e-004, 7.238e-004, 1.124e-002},
                           {85.000, 2.681e-004, 6.036e-004, 9.637e-003},
                           {90.000, 2.459e-004, 5.241e-004, 8.411e-003},
                           {95.000, 2.315e-004, 4.703e-004, 7.396e-003},
                           {100.000, 2.239e-004, 4.363e-004, 6.694e-003},
                           {105.000, 2.225e-004, 4.189e-004, 6.220e-003},
                           {110.000, 2.239e-004, 4.073e-004, 5.891e-003},
                           {115.000, 2.265e-004, 3.994e-004, 5.729e-003},
                           {120.000, 2.339e-004, 3.972e-004, 5.549e-003},
                           {125.000, 2.505e-004, 3.984e-004, 5.343e-003},
                           {130.000, 2.629e-004, 4.071e-004, 5.154e-003},
                           {135.000, 2.662e-004, 4.219e-004, 4.967e-003},
                           {140.000, 2.749e-004, 4.458e-004, 4.822e-003},
                           {145.000, 2.896e-004, 4.775e-004, 4.635e-003},
                           {150.000, 3.088e-004, 5.232e-004, 4.634e-003},
                           {155.000, 3.304e-004, 5.824e-004, 4.900e-003},
                           {160.000, 3.627e-004, 6.665e-004, 5.142e-003},
                           {165.000, 4.073e-004, 7.823e-004, 5.359e-003},
                           {170.000, 4.671e-004, 9.393e-004, 5.550e-003},
                           {175.000, 4.845e-004, 9.847e-004, 5.618e-003},
                           {180.000, 5.019e-004, 1.030e-003, 5.686e-003}}; // volumn scattering function lookup table: ang | VSF_clear | VSF_coast | VSF_turbid, from Petzold (1972) and Light and Water, Table 3.10

int NumElementsBXRA_table = sizeof(BXRA_table) / sizeof(BXRA_table[0]);
int NumElementsVSF_table = sizeof(VSF_table) / sizeof(VSF_table[0]);

namespace uw
{
    void Renderer::SetConfig(const std::string yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);

        scale_factor_ = config["scale_factor"].as<double>();
        scale_factor_bs_ = config["scale_factor_bs"].as<double>();
        render_back_scatter_ = config["render_back_scatter"].as<bool>();
        volumetric_max_depth_ = config["volumetric_max_depth"].as<double>();
        num_volumetric_slabs_ = config["num_volumetric_slabs"].as<int>();
        write_slab_ = config["write_slab"].as<bool>();
        slab_sampling_method_ = config["slab_sampling_method"].as<int>();
        depth_smooth_window_size_ = config["depth_smooth_window_size"].as<int>();
        refine_depth_ = config["refine_depth"].as<bool>();

        std::vector<unsigned int> img_size_vec = config["image_width_height"].as<std::vector<unsigned int>>();
        cv::Size img_size(img_size_vec[0], img_size_vec[1]);
        //    double field_of_view = config["field_of_view"].as<double>();
        std::vector<double> camera_intrinsic_matrix = config["camera_intrinsic_matrix"].as<std::vector<double>>();
        if (camera_intrinsic_matrix.size() == 1)
        {
            double field_of_view = camera_intrinsic_matrix[0];
            cam_.SetIntrinsics(img_size, field_of_view);
        }
        else if (camera_intrinsic_matrix.size() == 9)
        {
            Eigen::Matrix3d came_matrix; // = Eigen::Map<Eigen::Matrix3d>(camera_intrinsic_matrix.data());
            came_matrix << camera_intrinsic_matrix[0], camera_intrinsic_matrix[1], camera_intrinsic_matrix[2],
                camera_intrinsic_matrix[3], camera_intrinsic_matrix[4], camera_intrinsic_matrix[5],
                camera_intrinsic_matrix[6], camera_intrinsic_matrix[7], camera_intrinsic_matrix[8];
            cam_.SetIntrinsics(img_size, came_matrix);
        }
        else
        {
            std::cerr << "Input camera_intrinsic_matrix is in wrong format!!!" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::vector<double> white_balance_vec = config["white_balance"].as<std::vector<double>>();
        white_balance_ = Eigen::Map<Eigen::Vector3d>(white_balance_vec.data());
        cam_.SetWB(white_balance_);
        std::vector<double> attenuation_vec = config["water_attenuation_RGB"].as<std::vector<double>>();
        attenuation_ = Eigen::Map<Eigen::Vector3d>(attenuation_vec.data());
        vsf_type_ = config["vsf_type"].as<int>();
        std::vector<double> light_spectrum_vec = config["light_spectrum_RGB"].as<std::vector<double>>();
        light_spectrum_ = Eigen::Map<Eigen::Vector3d>(light_spectrum_vec.data());
        std::vector<double> light_ambient_vec = config["light_ambient_RGB"].as<std::vector<double>>();
        light_ambient_ = Eigen::Map<Eigen::Vector3d>(light_ambient_vec.data());
        light_type_ = config["light_RID_type"].as<int>();
        num_light_ = config["num_lights"].as<unsigned int>();
        std::vector<std::vector<double>> light_pos = config["light_positions_XYZ"].as<std::vector<std::vector<double>>>();
        std::vector<std::vector<double>> light_ori = config["light_orientations_RollPitchYaw"].as<std::vector<std::vector<double>>>();
        for (int i = 0; i < num_light_; i++)
        {
            Eigen::Vector3d l_pos, l_ori;
            l_pos = Eigen::Map<Eigen::Vector3d>(light_pos[i].data());
            l_ori = Eigen::Map<Eigen::Vector3d>(light_ori[i].data());
            Light light_buffer(l_pos, l_ori);
            light_buffer.SetSpecturm(light_spectrum_);
            light_buffer.SetAmbient(light_ambient_);
            light_buffer.SetRIDType(light_type_);
            lights_.emplace_back(light_buffer);
        }

        std::cerr << "Set Rendering Setups:" << std::endl
                  << "scale_factor: " << scale_factor_ << std::endl
                  << "scale_factor_bs: " << scale_factor_bs_ << std::endl
                  << "render_back_scatter: " << render_back_scatter_ << std::endl
                  << "volumetric_max_depth: " << volumetric_max_depth_ << std::endl
                  << "num_volumetric_slabs: " << num_volumetric_slabs_ << std::endl
                  << "write_slab: " << write_slab_ << std::endl
                  << "slab_sampling_method: " << slab_sampling_method_ << std::endl
                  << "depth_smooth_window_size: " << depth_smooth_window_size_ << std::endl
                  << "image_width&height: " << img_size << std::endl
                  << "focal_length: " << cam_.f() << std::endl
                  << "camera_matrix: " << cam_.K() << std::endl
                  << "camera_white_balance: " << white_balance_.transpose() << std::endl
                  << "number_of_lights: " << lights_.size() << std::endl
                  << "light_ambient_factor: " << light_ambient_.transpose() << std::endl
                  << "VSF_type: " << vsf_type_ << std::endl;
        for (int i = 0; i < num_light_; i++)
        {
            std::cerr << "Light " << i << ": position [" << lights_[i].pos().transpose() << "] direction [" << lights_[i].dir().transpose() << "]"
                      << " spectrum_RGB: [" << lights_[i].spectrum().transpose() << "] RID_type: " << lights_[i].RID_type() << std::endl;
        }

        if (render_back_scatter_)
        {
            vol_field_.Initialize(num_volumetric_slabs_, cam_.img_size());
            CalculateSampling();
            vol_field_.SetAtteCoeff(attenuation_);
        }
        write_uw_img_in_exr_ = config["write_uw_img_in_exr"].as<bool>();
        auto_iso_ = config["auto_iso"].as<bool>();
    }

    void Renderer::CalculateSampling()
    {
        int slab_numbers = num_volumetric_slabs_;
        double n = (double)slab_numbers;

        switch (slab_sampling_method_)
        {
        case 0:
        {
            double current_depth = 0.0;
            const double thickness = volumetric_max_depth_ / n;
            for (int i = 0; i < num_volumetric_slabs_; i++)
            {
                current_depth = current_depth + 0.5 * thickness;
                vol_field_.SetThickness(i, thickness);
                vol_field_.SetDepth(i, current_depth);
                current_depth = current_depth + 0.5 * thickness;
                std::cerr << "slab thickness: " << thickness << std::endl;
            }
            break;
        }
        case 1:
        {
            double current_depth = 0.0;
            for (int i = 0; i < num_volumetric_slabs_; i++)
            {
                double slab_i_thickness = 2.2 * volumetric_max_depth_ / exp(n) * pow(n, (double)(i)) / (double)factorial(i);
                current_depth = current_depth + 0.5 * slab_i_thickness;
                vol_field_.SetThickness(i, slab_i_thickness);
                vol_field_.SetDepth(i, current_depth);
                current_depth = current_depth + 0.5 * slab_i_thickness;
                std::cerr << "slab thickness: " << slab_i_thickness << std::endl;
            }

            double new_volumetric_max_depth = 0.0;
            for (int i = 0; i < num_volumetric_slabs_; i++)
            {
                new_volumetric_max_depth = new_volumetric_max_depth + vol_field_.GetThickness(i);
            }
            volumetric_max_depth_ = new_volumetric_max_depth;

            break;
        }
        }
    }

    void Renderer::RenderFS(double average_depth, cv::Mat &slab)
    {
        cv::Mat forwardSca_bs = cv::Mat(slab.size(), CV_64FC3, cv::Scalar(0.0, 0.0, 0.0));
        int GaussKernelWidth = (int)(average_depth / 2) * 2 + 1;
        cv::Size GaussKernelSize = cv::Size(GaussKernelWidth, GaussKernelWidth);

        cv::GaussianBlur(slab, forwardSca_bs, GaussKernelSize, 0, 0);
        slab = slab + forwardSca_bs;
    }

    void Renderer::ComputeSlabBS()
    {
        if (render_back_scatter_)
        {

            // rendering backscatter
            for (int i = 0; i < vol_field_.GetSlabNum(); i++)
            {
                cv::Mat slab_buffer(cam_.img_size(), CV_64FC3, cv::Scalar(0.0, 0.0, 0.0));
#pragma omp parallel for
                for (int r = 0; r < cam_.height(); r++)
                {
                    for (int c = 0; c < cam_.width(); c++)
                    {
                        Eigen::Vector3d pixel_2d_homo((double)c, (double)r, 1.0);
                        Eigen::Vector3d vec_cam2voxel = cam_.invK() * pixel_2d_homo;
                        vec_cam2voxel.normalize();
                        Eigen::Vector3d voxel_3d;
                        voxel_3d = vec_cam2voxel * vol_field_.GetDepth(i) / abs(vec_cam2voxel[2]); // origion of vector is at (0,0,0), so omitted
                        double d_cam2voxel = voxel_3d.norm();                                      // origion is at (0,0,0), so omitted
                        double cos_phi = vec_cam2voxel[2];

                        double R_voxel = 1.0;
                        double G_voxel = 1.0;
                        double B_voxel = 1.0;

                        double R_voxel_buffer = 0.0;
                        double G_voxel_buffer = 0.0;
                        double B_voxel_buffer = 0.0;

                        for (int n = 0; n < lights_.size(); n++)
                        {
                            int light_type = lights_[n].RID_type(); // apply for all lights
                            Eigen::Vector3d vec_light2voxel = voxel_3d - lights_[n].pos();
                            double d_light2voxel = vec_light2voxel.norm();
                            vec_light2voxel.normalize();

                            double cos_omega = (vec_cam2voxel[0] * vec_light2voxel[0] +
                                                vec_cam2voxel[1] * vec_light2voxel[1] +
                                                vec_cam2voxel[2] * vec_light2voxel[2]);
                            double omega = acos(cos_omega); // unit:RAD
                            double omega_deg = 180.0 - (omega / M_PI * 180.0);

                            double angle_from_central_axis = acos(lights_[n].dir()[0] * vec_light2voxel[0] +
                                                                  lights_[n].dir()[1] * vec_light2voxel[1] +
                                                                  lights_[n].dir()[2] * vec_light2voxel[2]) /
                                                             M_PI * 180.0; // unit: degree
                            double compute_buffer = get_vsf_value(omega_deg, vsf_type_) * light_RID(angle_from_central_axis, light_type) 
                                                    / d_light2voxel / d_light2voxel * vol_field_.GetThickness(i) * cos_phi * scale_factor_;  
                            // todo, need to further proof: replace 'cos_phi' by 'cos_beta/cos_phi/cos_phi' make more sense? beta is the angle between the ray voxel2light and inverse optical axis.
                            // cos_beta describe the energy on the unit area on voxel, one cos_phi describe the voxel area increase with phi, another cos_phi describe the ray travel distance increase in the voxel with phi: dist = thickness*cos_phi
                            R_voxel_buffer = R_voxel_buffer + exp(-vol_field_.GetAtteCoeff()[0] * d_light2voxel) * compute_buffer;
                            G_voxel_buffer = G_voxel_buffer + exp(-vol_field_.GetAtteCoeff()[1] * d_light2voxel) * compute_buffer;
                            B_voxel_buffer = B_voxel_buffer + exp(-vol_field_.GetAtteCoeff()[2] * d_light2voxel) * compute_buffer;
                        }

                        R_voxel = R_voxel_buffer * lights_[0].spectrum()[0] * exp(-vol_field_.GetAtteCoeff()[0] * d_cam2voxel); // only valid for the same type of light
                        G_voxel = G_voxel_buffer * lights_[0].spectrum()[1] * exp(-vol_field_.GetAtteCoeff()[1] * d_cam2voxel);
                        B_voxel = B_voxel_buffer * lights_[0].spectrum()[2] * exp(-vol_field_.GetAtteCoeff()[2] * d_cam2voxel);
                        slab_buffer.at<cv::Vec3d>(r, c) = cv::Vec3d(R_voxel, G_voxel, B_voxel); // in RGB order here !!!
                    }
                }

                RenderFS(vol_field_.GetDepth(i), slab_buffer);
                vol_field_.SetSlabVal(i, slab_buffer);

                std::cerr << "Slab " << i << " constructed." << std::endl;
            }
        }
    }

    void Renderer::AccumulateBS() // compute accumulated field
    {
        if (render_back_scatter_)
        {
            for (int i = 1; i < vol_field_.GetSlabNum(); i++)
            {
                vol_field_.SetSlabVal(i, vol_field_.GetSlab(i) + vol_field_.GetSlab(i - 1));
            }
            std::cerr << "Slab lookup table is constructed." << std::endl;
        }
    }
    
    void Renderer::WriteSlabs(std::string outpute_path)
    {
        if (render_back_scatter_)
        {
            for (int i = 0; i < vol_field_.GetSlabNum(); i++)
            {
                cv::Mat img_slab_temp;
                vol_field_.GetSlab(i).convertTo(img_slab_temp, CV_32FC3);

                std::string slab_temp_name = outpute_path + std::to_string(i) + ".exr";
                cv::imwrite(slab_temp_name, img_slab_temp);
            }
        }
    }

    cv::Mat Renderer::RenderUnderwater(const cv::Mat &img_air, cv::Mat &depth_map)
    {
        if (img_air.empty())
        {
            std::cerr << "input color image is empty." << std::endl;
            EXIT_FAILURE;
        }
        if (depth_map.empty())
        {
            std::cerr << "input depth map is empty." << std::endl;
            EXIT_FAILURE;
        }

        if (depth_map.channels() > 1)
        {
            std::cerr << "Depth image has " << depth_map.channels() << " channels, only the second channel is used. " << std::endl;
            cv::Mat bgr[3];
            cv::split(depth_map, bgr);

            depth_map.release();
            depth_map = bgr[1].clone();
        }

        if (refine_depth_)
        {
            // create depth inpainting mask
            cv::Mat depth_inpaint_mask = cv::Mat::zeros(depth_map.size(), CV_8U);
            for (int r = 0; r < depth_map.rows; r++)
            {
                for (int c = 0; c < depth_map.cols; c++)
                {
                    float depth_val = depth_map.at<float>(r, c);
                    if (depth_val < 0.1 || depth_val > volumetric_max_depth_)
                    {
                        depth_inpaint_mask.at<uchar>(r, c) = 255;
                    }
                    else
                        continue;
                }
            }
            std::string inpaint_mask_path = "./inpaint_mask.jpg";
            cv::imwrite(inpaint_mask_path, depth_inpaint_mask);

            // depth inpainting
            cv::Mat inpainted_depth;
            cv::inpaint(depth_map, depth_inpaint_mask, inpainted_depth, 5, cv::INPAINT_NS);
            std::string inpaint_depth_path = "./depth_inpainted.exr";
            depth_map = inpainted_depth.clone(); // update depth map to inpainted version
            cv::imwrite(inpaint_depth_path, depth_map);
        }
        else
        {
            for (int r = 0; r < depth_map.rows; r++)
            {
                for (int c = 0; c < depth_map.cols; c++)
                {
                    if (depth_map.at<float>(r, c) < 0.1)
                        depth_map.at<float>(r, c) = (float)volumetric_max_depth_;
                    else
                        continue;
                }
            }
        }

        cv::Mat img_direct(cam_.img_size(), CV_64FC3, cv::Scalar(0.0, 0.0, 0.0));
        cv::Mat img_bs(cam_.img_size(), CV_64FC3, cv::Scalar(0.0, 0.0, 0.0));

        uwc::DepthMap normal_estimator{depth_map};
        cv::Mat normals = normal_estimator.GetNormal(cam_.invK(), depth_smooth_window_size_);

        int light_type = lights_[0].RID_type();
        Eigen::Vector3d light_spectrum = lights_[0].spectrum(); // assume all lights have same specturm & RID

#pragma omp parallel for
        for (int r = 0; r < cam_.height(); r++)
        {
            for (int c = 0; c < cam_.width(); c++)
            {
                double depth_obj2cam = (double)depth_map.at<float>(r, c);
                Eigen::Vector3d p_2d_homo((double)c, (double)r, 1.0);
                Eigen::Vector3d vec_cam2obj = cam_.invK() * p_2d_homo;
                vec_cam2obj.normalize();
                Eigen::Vector3d p_3d;
                p_3d = vec_cam2obj * depth_obj2cam / abs(vec_cam2obj[2]); // origion is at (0,0,0), so omitted

                double d_cam2obj = p_3d.norm(); // origion is at (0,0,0), so omitted

                cv::Vec3b BGR = img_air.at<cv::Vec3b>(r, c);
                double R = (double)BGR[2];
                double G = (double)BGR[1];
                double B = (double)BGR[0];

                R = R * exp(-attenuation_[0] * d_cam2obj) * light_spectrum[0] * scale_factor_;
                G = G * exp(-attenuation_[1] * d_cam2obj) * light_spectrum[1] * scale_factor_;
                B = B * exp(-attenuation_[2] * d_cam2obj) * light_spectrum[2] * scale_factor_;

                double atten_buffer_R = 0.0;
                double atten_buffer_G = 0.0;
                double atten_buffer_B = 0.0;

                cv::Vec3d normalVec = normals.at<cv::Vec3d>(r, c);

                for (int i = 0; i < lights_.size(); i++)
                {
                    Eigen::Vector3d vec_light2obj = p_3d - lights_[i].pos();
                    double d_light2obj = vec_light2obj.norm(); // distance
                    vec_light2obj.normalize();
                    // check 3d point visibility from light
                    // if d_obj2light>deoth_lighjt, set d_obj2light=1000;

                    double cos_theta = (-normalVec[0] * vec_light2obj[0] - normalVec[1] * vec_light2obj[1] - normalVec[2] * vec_light2obj[2]);
                    double angle_from_central_axis = std::acos(lights_[i].dir()[0] * vec_light2obj[0] +
                                                               lights_[i].dir()[1] * vec_light2obj[1] +
                                                               lights_[i].dir()[2] * vec_light2obj[2]) / M_PI * 180.0; // unit: degree
                    double acc_buffer = light_RID(angle_from_central_axis, light_type) / d_light2obj / d_light2obj;

                    if (cos_theta < 0.0)
                    { // no lights hit this point, only ambient
                        atten_buffer_R = atten_buffer_R + exp(-attenuation_[0] * d_light2obj) * lights_[i].ambient()[0] * acc_buffer;
                        atten_buffer_G = atten_buffer_G + exp(-attenuation_[1] * d_light2obj) * lights_[i].ambient()[1] * acc_buffer;
                        atten_buffer_B = atten_buffer_B + exp(-attenuation_[2] * d_light2obj) * lights_[i].ambient()[2] * acc_buffer;
                    }
                    else
                    { // ambient + diffuse
                        atten_buffer_R = atten_buffer_R + exp(-attenuation_[0] * d_light2obj) * (cos_theta * acc_buffer + lights_[i].ambient()[0] * acc_buffer);
                        atten_buffer_G = atten_buffer_G + exp(-attenuation_[1] * d_light2obj) * (cos_theta * acc_buffer + lights_[i].ambient()[1] * acc_buffer);
                        atten_buffer_B = atten_buffer_B + exp(-attenuation_[2] * d_light2obj) * (cos_theta * acc_buffer + lights_[i].ambient()[2] * acc_buffer);
                    }
                }

                R = R * atten_buffer_R;
                G = G * atten_buffer_G;
                B = B * atten_buffer_B;

                img_direct.at<cv::Vec3d>(r, c) = cv::Vec3d(R, G, B); // in RGB order
            }
        }

        RenderFS(normal_estimator.mean(), img_direct); // add forward scattering

        if (render_back_scatter_)
        {
#pragma omp parallel for
            for (int r = 0; r < cam_.height(); r++)
            {
                for (int c = 0; c < cam_.width(); c++)
                {
                    double depth_obj2cam = (double)depth_map.at<float>(r, c);

                    if (depth_obj2cam > volumetric_max_depth_)
                    {
                        depth_obj2cam = volumetric_max_depth_;
                    }

                    double R_bs = 0.0;
                    double G_bs = 0.0;
                    double B_bs = 0.0;
                    if (depth_obj2cam >= volumetric_max_depth_ || depth_obj2cam <= 0.0) // <=0.0 means only water in the scene, no depth valuse
                    {
                        int depthID = vol_field_.GetSlabNum() - 1;
                        R_bs = vol_field_.GetSlab(depthID).at<cv::Vec3d>(r, c)[0];
                        G_bs = vol_field_.GetSlab(depthID).at<cv::Vec3d>(r, c)[1];
                        B_bs = vol_field_.GetSlab(depthID).at<cv::Vec3d>(r, c)[2];
                    }
                    else
                    {
                        cv::Vec3d bs_RGB = interpolate_bs(r, c, depth_obj2cam);
                        R_bs = bs_RGB[0];
                        G_bs = bs_RGB[1];
                        B_bs = bs_RGB[2];
                    }

                    img_bs.at<cv::Vec3d>(r, c) = cv::Vec3d(R_bs, G_bs, B_bs);
                }
            }
        }

        img_direct = img_direct + img_bs * scale_factor_bs_;

        if (write_uw_img_in_exr_)
        {
            cv::Mat img_new_32F;
            img_direct.convertTo(img_new_32F, CV_32FC3);
            cv::imwrite("./img_uw_EXR.exr", img_new_32F);
        }

        return img_direct;
    }

    void Renderer::WriteDoubleMatTo8Bit(const cv::Mat &double_mat, std::string &output_name)
    {
        cv::Mat img_new(double_mat.size(), CV_8UC3);

        if (auto_iso_)
        {
            double min;
            double max;
            cv::minMaxIdx(double_mat, &min, &max);
            double_mat.convertTo(img_new, CV_8UC3, 255.0 / (max - min), -min);
        }
        else
        {
#pragma omp parallel for
            for (int r = 0; r < double_mat.rows; r++)
            {
                for (int c = 0; c < double_mat.cols; c++)
                {
                    cv::Vec3d color = double_mat.at<cv::Vec3d>(r, c);

                    color[0] = color[0] * white_balance_[0];
                    color[1] = color[1] * white_balance_[1];
                    color[2] = color[2] * white_balance_[2];

                    if (color[0] >= 255.0)
                    {
                        color[0] = 255.0;
                    }
                    if (color[1] >= 255.0)
                    {
                        color[1] = 255.0;
                    }
                    if (color[2] >= 255.0)
                    {
                        color[2] = 255.0;
                    }

                    cv::Vec3b BGR_new((uchar)color[2], (uchar)color[1], (uchar)color[0]); // in BGR order for opencv imwrite

                    img_new.at<cv::Vec3b>(r, c) = BGR_new;
                }
            }
        }

        cv::imwrite(output_name, img_new);
    }

    void Renderer::RenderFogModel(const cv::Mat &color_img, const cv::Mat &depth_img)
    {
        Eigen::Vector3d attenuation(0.37, 0.044, 0.035);  // RGB
        Eigen::Vector3d water_color(110.0, 137.0, 212.0); // RGB
        cv::Mat output(color_img.size(), color_img.type());

#pragma omp parallel for
        for (int r = 0; r < color_img.rows; r++)
        {
            for (int c = 0; c < color_img.cols; c++)
            {
                double depth_obj2cam = (double)depth_img.at<float>(r, c);
                //            std::cerr << depth_obj2cam << std::endl;
                Eigen::Vector3d p_2d_homo((double)c, (double)r, 1.0);
                Eigen::Vector3d vec_cam2obj = cam_.invK() * p_2d_homo;
                vec_cam2obj.normalize();
                Eigen::Vector3d p_3d;
                p_3d = vec_cam2obj * depth_obj2cam / abs(vec_cam2obj[2]); // origion is at (0,0,0), so omitted

                double d_cam2obj = p_3d.norm(); // origion is at (0,0,0), so omitted

                cv::Vec3b BGR = color_img.at<cv::Vec3b>(r, c);
                double R = (double)BGR[2];
                double G = (double)BGR[1];
                double B = (double)BGR[0];

                R = R * exp(-attenuation[0] * d_cam2obj) + water_color[0] * (1.0 - exp(-attenuation[0] * d_cam2obj));
                G = G * exp(-attenuation[1] * d_cam2obj) + water_color[1] * (1.0 - exp(-attenuation[1] * d_cam2obj));
                B = B * exp(-attenuation[2] * d_cam2obj) + water_color[2] * (1.0 - exp(-attenuation[2] * d_cam2obj));

                output.at<cv::Vec3b>(r, c) = cv::Vec3b((uchar)B, (uchar)G, (uchar)R); // BGR!
            }
        }
        cv::imwrite("../testset_bomb/fog_model_result.jpg", output);
    }

    void Renderer::WriteBSField(std::string path)
    {
        if (render_back_scatter_)
        {
            std::string bs_field_name = path + "bs_field.exr";
            cv::Mat img_bs_field;
            vol_field_.GetSlab(vol_field_.GetSlabNum() - 1).convertTo(img_bs_field, CV_32FC3);
            cv::imwrite(bs_field_name, img_bs_field);

            img_bs_field = img_bs_field * scale_factor_bs_;

            cv::Mat img_bs_field_double;
            img_bs_field.convertTo(img_bs_field_double, CV_64FC3);
            std::string bs_8bit_name = path + "bs_field.png";
            WriteDoubleMatTo8Bit(img_bs_field_double, bs_8bit_name);
        }
    }

    void GenerateBSPlot(std::string output_file, double max_dist, double sampling_dist)
    {
        int vsf_type = 1;
        Eigen::Vector3d atten_coeff(0.4, 0.071, 0.06); // Jerlov type II
        Eigen::Vector3d light_spectrum(0.25, 0.35, 0.4);
        /// generate figures jerlov
        std::ofstream curve(output_file, std::ios_base::out);
        for (double depth = 0.1; depth < (max_dist + 0.1 * sampling_dist); depth = depth + sampling_dist)
        {
            std::cerr << depth << std::endl;
            Eigen::Vector3d vec_cam2voxel(0.0, 0.0, depth);
            vec_cam2voxel.normalize();
            Eigen::Vector3d p_3d(0.0, 0.0, depth);
            double d_cam2voxel = depth; // origion is at (0,0,0), so omitted

            double R_voxel = 1.0;
            double G_voxel = 1.0;
            double B_voxel = 1.0;

            double R_voxel_buffer = 0.0;
            double G_voxel_buffer = 0.0;
            double B_voxel_buffer = 0.0;

            Eigen::Vector3d light_position(1.0, 1.0, 0.0);
            Eigen::Vector3d vec_light2voxel = p_3d - light_position;
            double d_light2voxel = vec_light2voxel.norm();
            vec_light2voxel.normalize();

            double cos_omega = (vec_cam2voxel[0] * vec_light2voxel[0] +
                                vec_cam2voxel[1] * vec_light2voxel[1] +
                                vec_cam2voxel[2] * vec_light2voxel[2]);
            double omega = acos(cos_omega); // unit:RAD
            double omega_deg = 180.0 - (omega / M_PI * 180.0);

            double compute_buffer = get_vsf_value(omega_deg, vsf_type) / d_light2voxel / d_light2voxel;

            R_voxel_buffer = R_voxel_buffer + exp(-atten_coeff[0] * d_light2voxel) * compute_buffer;
            G_voxel_buffer = G_voxel_buffer + exp(-atten_coeff[1] * d_light2voxel) * compute_buffer;
            B_voxel_buffer = B_voxel_buffer + exp(-atten_coeff[2] * d_light2voxel) * compute_buffer;

            R_voxel = R_voxel_buffer * light_spectrum[0] * exp(-atten_coeff[0] * d_cam2voxel);
            G_voxel = G_voxel_buffer * light_spectrum[1] * exp(-atten_coeff[1] * d_cam2voxel);
            B_voxel = B_voxel_buffer * light_spectrum[2] * exp(-atten_coeff[2] * d_cam2voxel);

            curve << R_voxel << " " << G_voxel << " " << B_voxel << "\n";
        }
        curve.close();
    }

    double light_RID(double &angle, int &light_type) // unit: degree
    {
        if (light_type == 0) // 0: gaussian curve
        {
            if (angle < 0.0 || angle > 90.0)
            {
                return 0.0;
            }
            else
            {
                return (exp(-angle * angle / (2.0 * 35.0 * 35.0)));
            }
        }

        if (light_type == 1) // 1: real measurement
        {
            if (angle < 0.0 || angle > 90.0)
            {
                return 0.0;
            }
            else
            {
                int tableIDX = 0;
                for (int idx = 0; idx < NumElementsBXRA_table; idx++)
                {
                    if (angle > BXRA_table[idx][0])
                    {
                        tableIDX++;
                    }
                    else
                        break;
                }
                if (tableIDX > 0)
                {
                    tableIDX--;
                }
                double factor;
                double startAng, startVal, endAng, endVal;
                startAng = BXRA_table[tableIDX][0];
                startVal = BXRA_table[tableIDX][1];

                if (tableIDX >= NumElementsBXRA_table)
                {
                    factor = 0.0;
                }
                else
                {
                    endAng = BXRA_table[tableIDX + 1][0];
                    endVal = BXRA_table[tableIDX + 1][1];
                    factor = startVal + abs(angle - startAng) * (endVal - startVal) / (endAng - startAng);
                }

                return factor;
            }
        }

        else
        {
            std::cerr << "unknown light RID type." << std::endl;
            EXIT_FAILURE;
            return 0;
        }
    }

    double get_vsf_value(double &angle, int vsf_type)
    {
        //   vsf_type  1:VSF_clear, 2:VSF_coast,  3:VSF_turbid, 4: Henyey-Greenstein Phase Function
        double factor = 0.0;
        if(vsf_type<4){
            int tableIDX = 0;
        for (int idx = 0; idx < NumElementsVSF_table; idx++)
        {
            if (angle > VSF_table[idx][0])
            {
                tableIDX++;
            }
            else
                break;
        }
        if (tableIDX > 0)
        {
            tableIDX--;
        }
        
        double startAng, startVal, endAng, endVal;
        startAng = VSF_table[tableIDX][0];
        startVal = VSF_table[tableIDX][vsf_type];

        if (tableIDX >= NumElementsVSF_table)
        {
            factor = 0.0;
        }
        else
        {
            endAng = VSF_table[tableIDX + 1][0];
            endVal = VSF_table[tableIDX + 1][vsf_type];
            factor = startVal + abs(angle - startAng) * (endVal - startVal) / (endAng - startAng);
        }

        }

        else if(vsf_type==4){
            double g = -0.3; // Henyey-Greenstein parameter, range [-1.0, 1.0]
            double temp = 1.0 + g*g + 2.0*g*std::cos(angle / 180.0 * M_PI);
            factor = 1.0 / 4.0 / M_PI * (1.0 - g*g) / (temp*std::sqrt(temp));
        }
        else{
            std::cout << "Unknown VSF type! " << std::endl;
            EXIT_FAILURE;
            return 0;
        }

        return factor;
    }

    void GenerateRandomFlatSeafloor(int num_img, cv::Size img_size, double fov, std::string output_path) // this will generate depth and color images without water effect (only lighting effect)
    {
        double focal = img_size.width / 2.0 / tan(fov / 2.0);
        Eigen::Matrix3d KMat_cam;
        KMat_cam << focal, 0.0, (double)img_size.width / 2.0,
            0.0, focal, (double)img_size.height / 2.0,
            0.0, 0.0, 1.0;
        Eigen::Matrix3d KMat_cam_inv = KMat_cam.inverse();

        cv::Mat colorImage(img_size, CV_8UC3);
        colorImage = cv::Scalar(150, 150, 150); // color image are set to the same color
        cv::Mat depthImage(img_size, CV_32FC1);

        for (int i = 0; i < num_img; i++)
        {
            std::string colorPath = output_path + "simple_" + std::to_string(i) + ".png";
            std::string depthPath = output_path + "simple_" + std::to_string(i) + ".exr";

            std::random_device rd;                                                       // obtain a random number from hardware
            std::mt19937 eng(rd());                                                      // seed the generator
            std::uniform_real_distribution<double> distr_depth(0.5, 3.5);                // define the range of depth
            std::uniform_real_distribution<double> distr_roll(-M_PI / 9.0, M_PI / 9.0);  // define the rotation range along X axis
            std::uniform_real_distribution<double> distr_pitch(-M_PI / 9.0, M_PI / 9.0); // define the rotation range along Y axis

            //        // plane equation nxX+nyY+nz(Z-d)=0
            double depth = distr_depth(eng);
            double rot_roll = distr_roll(eng);
            double rot_pitch = distr_pitch(eng);
            Eigen::Vector3d n_vec(0.0, 0.0, 1.0);
            Eigen::Matrix3d rotation;
            rotation = Eigen::AngleAxisd(rot_roll, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(rot_pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ());
            n_vec = rotation * n_vec;

            // double cx = (double)imgSize.width/2.0;
            // double cy = (double)imgSize.height/2.0;
            Eigen::Vector3d planeP(0.0, 0.0, depth);
            Eigen::Vector3d cameraCenter(0.0, 0.0, 0.0);
            for (int r = 0; r < colorImage.rows; r++)
            {
                for (int c = 0; c < colorImage.cols; c++)
                {
                    Eigen::Vector3d p_2d_homo((double)c, (double)r, 1.0);
                    if (abs(p_2d_homo[0]) < 0.000000000001 && abs(p_2d_homo[1]) < 0.000000000001)
                    {
                        depthImage.at<float>(r, c) = (float)depth;
                    }
                    else
                    {
                        Eigen::Vector3d pixel_ray = KMat_cam_inv * p_2d_homo;
                        Eigen::Vector3d P = ray_plane_intersection(pixel_ray, cameraCenter, n_vec, planeP);
                        depthImage.at<float>(r, c) = (float)P[2];
                    }
                }
            }
            cv::imwrite(colorPath, colorImage);
            cv::imwrite(depthPath, depthImage);
        }
    }

    Eigen::Vector3d ray_plane_intersection(Eigen::Vector3d &rayVector, Eigen::Vector3d &rayPoint, Eigen::Vector3d &planeNormal, Eigen::Vector3d &planePoint) // algorithm from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#C.2B.2B
    {
        Eigen::Vector3d diff = rayPoint - planePoint;
        double prod1 = diff.dot(planeNormal);
        double prod2 = rayVector.dot(planeNormal);
        double prod3 = prod1 / prod2;
        return rayPoint - rayVector * prod3;
    }

    void RenderPlanImage(int render_img_num, std::string outpu_path)
    {
        // this part is only for generating simple(plane) dataset
        cv::Size img_size(800, 800);
        GenerateRandomFlatSeafloor(render_img_num, img_size, 1.3962634, outpu_path);
    }

    int factorial(int n)
    {
        return (n == 1 || n == 0) ? 1 : n * factorial(n - 1);
    }

    cv::Vec3d Renderer::interpolate_bs(int row, int col, const double &depth_val)
    {
        cv::Vec3d output;
        double d = depth_val;

        if (d >= volumetric_max_depth_)
        {
            output = vol_field_.GetSlab(vol_field_.GetSlabNum() - 1).at<cv::Vec3d>(row, col);
        }
        else
        {
            int tableIDX = 0;
            for (int idx = 0; idx < vol_field_.GetSlabNum(); idx++)
            {
                d = d - vol_field_.GetThickness(idx);
                if (d > 0.0)
                {
                    tableIDX++;
                }
                else
                    break;
            }

            if (tableIDX == 0)
            {
                output = depth_val / vol_field_.GetThickness(0) * vol_field_.GetSlab(0).at<cv::Vec3d>(row, col);
            }
            else
            {
                double mod = depth_val;
                for (int id = 0; id < tableIDX; id++)
                {
                    mod = mod - vol_field_.GetThickness(id);
                }
                cv::Vec3d startVal = vol_field_.GetSlab(tableIDX - 1).at<cv::Vec3d>(row, col);
                cv::Vec3d endVal = vol_field_.GetSlab(tableIDX).at<cv::Vec3d>(row, col);
                cv::Vec3d diff = endVal - startVal;
                output = startVal + diff * mod / vol_field_.GetThickness(tableIDX);
            }
        }

        return output;
    }

    void ExrImageAmplify(std::string exr_filename, float times)
    {
        cv::Mat img = cv::imread(exr_filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        img = img * times;
        std::string path = exr_filename.substr(0, exr_filename.size() - 4);
        std::string extension = exr_filename.substr(exr_filename.size() - 4, exr_filename.size() - 1);
        std::string exr_filename_new = path + "_X" + std::to_string(times) + extension;
        cv::imwrite(exr_filename_new, img);
    }

}
