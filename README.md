# **Deep Sea Robotic Imaging Simulator**

**Project Homepage:** <https://www.geomar.de/en/omv-research/robotic-imaging-simulator>

**GitHub page:** <https://github.com/omv-ysong/DeepSeaRenderer>

## Publication
```
@inproceedings{song2021deep,
  title={Deep Sea Robotic Imaging Simulator},
  author={Song, Yifan and Nakath, David and She, Mengkun and Elibol, Furkan and K{\\"o}ser, Kevin},
  booktitle={Pattern Recognition. ICPR International Workshops and Challenges},
  year={2021},
  publisher={Springer},
  pages={375--389},
  doi={https://doi.org/10.1007/978-3-030-68790-8_29}
}
```

**New features are added to handle real world scene with imperfect depth maps.** They have been used in: 

```
@article{song2022virtually,
  title={Virtually throwing benchmarks into the ocean for deep sea photogrammetry and image processing evaluation},
  author={Song, Yifan and She, Mengkun and K{\"o}ser, Kevin},
  journal={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={V-4-2022},
  pages={353--360},
  year={2022},
  doi={https://doi.org/10.1007/978-3-030-68790-8_29}
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
## running test on real world scene RGB-D images
(Test images are in *DeepSeaRenderer/testset_middlebury/Adirondack_perfect/*). You can call ./testRenderer with specified config, image and depth list files (must 3 input files in fixed order, otherwise the default files will be loaded.): 
```
cd bin
./testRenderer ../testset_middlebury/Adirondack_perfect/config.yaml ../testset_middlebury/Adirondack_perfect/img_lists.txt ../testset_middlebury/Adirondack_perfect/depth_lists.txt
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
```

3. Pre-compute 3D backscatter lookup table:

```
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
renderer.WriteDoubleMatTo8Bit(new_img_double, output_name);
```

## Description of parameters in the config file

| Parameter Name                  | Type                    | Description                                                                                                                                                                                                   | Example                                                    |
|---------------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| scale_factor                    | float                   | control the global brightness, similar to the ISO or exposure time of the camera                                                                                                                             | 8.0                                                        |
| scale_factor_bs                 | float                   | control the strength of the backscatter effect, usually set to 1000 times  to the scale_factor. Increase the factor will make stronger  backscatter in the rendered images.                                   | 8000.0                                                     |
| render_back_scatter             | bool                    | render backscatter effect in the output images, this will pre-render  the 3D backscatter lookup table.                                                                                                        | true                                                       |
| volumetric_max_depth            | float                   | max depth [m] of the 3D backscatter lookup table (recommend values: 8-10).                                                                                                                                    | 8.0                                                        |
| num_volumetric_slabs            | int                     | number of slabs in the 3D backscatter lookup table.                                                                                                                                                           | 10                                                         |
| write_slab                      | bool                    | save all slabs in the 3D backscatter lookup table as EXR images.                                                                                                                                              | false                                                      |
| slab_sampling_method            | int                     | slab sampling method, 0: EQUAL_DISTANCE, 1: ADAPTIVE                                                                                                                                                          | 1                                                          |
| image_width_height              | [int, int]              | image width and height in pixels. (Don't miss the square bracket: [])                                                                                                                                         | [800, 600]                                                 |
| camera_intrinsic_matrix         | [float, float, ... (9 elements)]   | camera intrinsic matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1], if only one element is given will be read as camera field of view in radian.                                                                                                        | [600.0, 0.0, 400.0, 0.0, 600.0, 400.0, 0.0, 0.0, 1.0]                                                   |
| white_balance                   | [float, float, float]   | camera white balance in R, G, B.                                                                                                                                                                              | [2.2, 1.0, 1.4]                                            |
| water_attenuation_RGB           | [float, float, float]   | water attenuation parameters in R, G, B. (unit: m-¹)                                                                                                                                                          | [0.37, 0.044, 0.035]                                       |
| vsf_type                  | int                     | lVolume Scattering Function, 1:VSF_clear, 2:VSF_coast,  3:VSF_turbid (1-3 from Pezolds' measurements), 4:H-G.                            | 3                                                          |
| light_spectrum_RGB              | [float, float, float]   | light spectrum in R, G, B. (relative value, range in [0.0, 1.0])                                                                                                                                              | [0.25, 0.35, 0.4]                                          |
| light_ambient_RGB              | [float, float, float]   | ambient light component (Phong shading) in R,G,B order. (range in [0.0, 1.0])                                                                                                                                              | [0.2, 0.2, 0.2]                                          |
| light_RID_type                  | int                     | light RID type, 0: gaussian, 1: lab measurement.                                                                                                                                                               | 0                                                          |
| num_lights                      | int                     | number of lights.                                                                                                                                                                                             | 3                                                          |
| light_positions_XYZ             | [[float, float, float]] | relative position [in meter] of all lights refer to the camera, number of XYZ coordinates should fit to the number of lights. (each square bracket contains XYZ for each light: [X, Y, Z], another square bracket includes all the lights.) | [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]       |
| light_orientations_RollPitchYaw | [[float, float, float]] | relative rotation angles [radian] of all lights in Roll, Pitch, Yaw,  initial orientation is the camera viewing direction,  number of RollPitchYaw angles should fit to the number of lights.                 | [[0.0, -0.785, 0.0], [0.785, 0.0, 0.0], [0.0, 0.785, 0.0]] |
| write_uw_img_in_exr             | bool                    | save float images in exr format.                                                                                                                                                                              | false                                                      |
| depth_smooth                                        _window_size                  | int                     | kernal size of the median filter for smoothing normal eages, 0 means no smoothing, can only be 0, 3, 5.                                                                                                                                                               | 3                                                          |
| refine_depth             | bool                    | inpaint incomplete depth maps if needed                                                                                                                                                                              | false                                                      |
| auto_iso                        | bool                    | automatically adjust the brightness of the output images, similar to auto ISO + auto WB settings in digital camera, output color may change because it adjust channels separately.                            | false                                                      |
