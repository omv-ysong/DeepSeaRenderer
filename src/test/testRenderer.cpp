#include "../depth.h"
#include "../rendering.h"
#include <opencv2/opencv.hpp>
#include <fstream>

int main()
{
    std::string images_list = "../test_dataset/img_lists.txt";
    std::string detph_list = "../test_dataset/depth_lists.txt";
    std::string config_filename = "../test_dataset/config.yaml";

    std::ifstream images_in(images_list, std::ios_base::in);
    std::ifstream detph_in(detph_list, std::ios_base::in);
    uw::Renderer renderer;
    renderer.SetConfig(config_filename);
    renderer.ComputeSlabBS();
    //renderer.WriteSlabs("../test_data/test0/");
    renderer.AccumulateBS();

    std::string image_path, depth_path;
    std::cerr << "Start rendering... " << std::endl;
    while(std::getline(images_in, image_path) && std::getline(detph_in, depth_path))
    {
        std::cerr << image_path << " " << depth_path << std::endl;
        std::string output_name = image_path.substr(0, image_path.size()-4) + "_uw.jpg";

        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
            cv::Mat dMap_cam = cv::imread(depth_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
            dMap_cam = dMap_cam*1.5;
            for(int r=0; r<dMap_cam.rows; r++)
            {
                for(int c=0; c<dMap_cam.cols; c++)
                {
                    if(dMap_cam.at<float>(r,c)<0.1 | dMap_cam.at<float>(r,c)>10.0)
                        dMap_cam.at<float>(r,c) = 1.0f;
                    else
                        continue;
                }
            }
            if(img.empty())
            {
                std::cerr << "input color image is empty." << std::endl;
                system("pause");
                exit(0);
            }
            if(dMap_cam.empty())
            {
                std::cerr << "input depth map is empty." << std::endl;
                system("pause");
                exit(0);
            }

        cv::Mat new_img_double = renderer.RenderUnderwater(img, dMap_cam);
        renderer.ConvertDoubleMatTo8Bit(new_img_double, output_name);
    }

    images_in.close();
    detph_in.close();


    std::cout << "Rendering Finish." << std::endl;

    return 0;
}

