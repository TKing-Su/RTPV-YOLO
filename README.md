# RTPV-YOLO: Real-Time Photovoltaic Defect Detection with UAV-Based Thermal and RGB Imaging

**RTPV-YOLO** is a novel algorithm designed for real-time photovoltaic (PV) defect detection using Unmanned Aerial Vehicles (UAVs). The algorithm addresses traditional challenges in detecting small hot spots in thermal images and irregularly shaped obstructions (e.g., bird droppings or leaves) in RGB images. It integrates several advanced modules for enhanced accuracy and efficiency in detecting PV defects.

## Demo

Here is a demo of the algorithm detecting defects in real-time:

![Demo GIF](examples/hotspot-detect.gif)

## Raw Data Collection

The drone collects raw video data of photovoltaic (PV) panels during its inspection. These videos are crucial for training and evaluating detection models. The raw data includes both thermal and RGB footage captured at various altitudes.

You can access and download the raw video data from the following Baidu Netdisk link:

[Baidu Netdisk - Raw Video Data](https://pan.baidu.com/s/1yPEjzDxNqisjyBJwAx3cUw)

### How to Access:
1. Click on the link above.
2. Enter the provided access code (4019).
3. Download the video files for further use.

## Features

- **Real-Time Detection**: The algorithm enables efficient, real-time detection of PV defects using UAVs.
- **Advanced Modules**: Utilizes **Spatial to Depth Convolution (SPD-Conv)** and **Convolutional Block Attention Module (CBAM)** to enhance detection performance for small and irregularly shaped defects.
- **Reparameterization Adaptive Kernel Convolution Module (RepAKCM)**: This innovative module reduces computational load while improving feature extraction and fusion, allowing for better performance on resource-constrained devices like UAVs.

## Performance

The **RTPV-YOLO** algorithm has been evaluated on a custom-built PV defect dataset and achieved a **mean Average Precision (mAP)** of **91%**, outperforming comparable models in terms of both accuracy and computational efficiency.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the development team at **Wuhan University of Technology** for their contribution to the project.
- Special thanks to the open-source community for the libraries and tools that made this project possible.
