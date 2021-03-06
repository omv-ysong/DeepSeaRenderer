# **Deep Sea Robotic Imaging Simulator**

**Project Homepage:** <https://www.geomar.de/en/omv-research/robotic-imaging-simulator>

**GitHub page:** <https://github.com/omv-ysong/DeepSeaRenderer>

## Publication
```
@inproceedings{song2020deep,
  title={Deep Sea Robotic Imaging Simulator},
  author={Song, Yifan and Nakath, David and She, Mengkun and Elibol, Furkan and K{\\"o}ser, Kevin},
  booktitle={Proceedings of the Computer Vision for Automated Analysis of Underwater Imagery Workshop (CVAUI)},
  year={2020},
  organization={Springer}
}
```

## Install dependencies from the default Ubuntu repositories

```
sudo apt-get install git cmake libeigen3-dev libyaml-cpp-dev libopencv-dev
```

## Configure and compile DeepSeaRenderer

```
git clone https://github.com/omv-ysong/DeepSeaRenderer.git
cd DeepSeaRenderer
cmake .
make
```

## running test
(Test images are in *DeepSeaRenderer/testset_bomb/*)
```
cd bin
./testRenderer
```

**If you want to make a quick test on your own imags**, the easiest way is:

1. Set proper parameters according to your requirements in the *DeepSeaRenderer/testset_bomb/config.yaml* file;
2. Replace the image and depth path in *img_lists.txt* and *depth_lists.txt* to your images. (Image and depth names should in the same order.) run:

```
cd bin
./testRenderer
```

## How to integrate the code to your own project

1. Set proper parameters in the *config.yaml* file;
2. Initialize renderer;

```
#include "src/depth.h"
#include "src/rendering.h"
#include <opencv2/opencv.hpp>

...

uw::Renderer renderer;
renderer.SetConfig(config_filename);
3. Pre-compute 3D backscatter lookup table:
renderer.ComputeSlabBS();
renderer.AccumulateBS();
```

4. Get in-air texture (cv::Mat, CV_8UC3) and depth (cv::Mat, CV_32FC1), example:

```
cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
cv::Mat dMap = cv::imread(depth_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
```

5. Generate underwater image.

```
cv::Mat new_img_double = renderer.RenderUnderwater(img, dMap);
cv::Mat new_img = renderer.ConvertDoubleMatTo8Bit(new_img_double);
cv::imwrite(output_name, new_img);
```

## Description of parameters in the config file

| Parameter Name                  | Type                    | Description                                                                                                                                                                                                   | Example                                                    |
|---------------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| scale_factor                    | float                   | control the global brightness.                                                                                                                                                                                | 8.0                                                        |
| scale_factor_bs                 | float                   | control the strength of the backscatter effect, usually set to 1000 times  to the scale_factor. Increase the factor will make stronger  backscatter in the rendered images.                                   | 8000.0                                                     |
| render_back_scatter             | bool                    | render backscatter effect in the output images, this will pre-render  the 3D backscatter lookup table.                                                                                                        | true                                                       |
| volumetric_max_depth            | float                   | max depth [m] of the 3D backscatter lookup table (recommend values: 8-10).                                                                                                                                    | 8.0                                                        |
| num_volumetric_slabs            | int                     | number of slabs in the 3D backscatter lookup table.                                                                                                                                                           | 10                                                         |
| write_slab                      | bool                    | save all slabs of the 3D backscatter lookup table in EXR images.                                                                                                                                              | false                                                      |
| slab_sampling_method            | int                     | slab sampling method, 0: EQUAL_DISTANCE, 1: ADAPTIVE                                                                                                                                                          | 1                                                          |
| image_width_height              | [int, int]              | image width and height in pixels. (Don't miss the square bracket: [])                                                                                                                                         | [800, 600]                                                 |
| field_of_view                   | float                   | camera field of view in radian.                                                                                                                                                                               | 1.57                                                       |
| white_balance                   | [float, float, float]   | camera white balance in R, G, B.                                                                                                                                                                              | [2.2, 1.0, 1.4]                                            |
| water_attenuation_RGB           | [float, float, float]   | water attenuation parameters in R, G, B. (unit: m-¹)                                                                                                                                                          | [0.37, 0.044, 0.035]                                       |
| light_spectrum_RGB              | [float, float, float]   | light spectrum in R, G, B. (relative value, range in [0.0, 1.0])                                                                                                                                              | [0.25, 0.35, 0.4]                                          |
| light_RID_type                  | int                     | ight RID type, 0: gaussian, 1: lab measurement.                                                                                                                                                               | 0                                                          |
| num_lights                      | int                     | number of lights.                                                                                                                                                                                             | 3                                                          |
| light_positions_XYZ             | [[float, float, float]] | relative position of all lights,  number of XYZ coordinates should fit to the number of lights. (each square bracket contains XYZ for each light: [X, Y, Z], another square bracket includes all the lights.) | [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]       |
| light_orientations_RollPitchYaw | [[float, float, float]] | relative rotation angles [radian] of all lights in Roll, Pitch, Yaw,  initial orientation is the camera viewing direction,  number of RollPitchYaw angles should fit to the number of lights.                 | [[0.0, -0.785, 0.0], [0.785, 0.0, 0.0], [0.0, 0.785, 0.0]] |
| write_uw_img_in_exr             | bool                    | save float images in exr format.                                                                                                                                                                              | false                                                      |
| auto_iso                        | bool                    | automatically adjust the brightness of the output images, similar to auto ISO + auto WB settings in digital camera, output color may change because it adjust channels separately.                            | false                                                      |
