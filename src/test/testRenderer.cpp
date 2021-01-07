#include "../depth.h"
#include "../rendering.h"
#include <opencv2/opencv.hpp>
#include <fstream>

int main()
{
    std::string images_list = "../testset_bomb/img_lists.txt";
    std::string detph_list = "../testset_bomb/depth_lists.txt";
    std::string config_filename = "../testset_bomb/config.yaml";

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

        cv::Mat new_img_double = renderer.RenderUnderwater(img, dMap_cam);
        renderer.ConvertDoubleMatTo8Bit(new_img_double, output_name);
    }

    images_in.close();
    detph_in.close();


    std::cout << "Rendering Finish." << std::endl;

    return 0;
}

