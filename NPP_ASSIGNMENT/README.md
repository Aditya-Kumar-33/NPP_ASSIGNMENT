# laplaceFilterNPP

### Description

The laplaceFilterNPP is a CUDA-based sample that demonstrates how to use the NVIDIA Performance Primitives (NPP) library to apply a Laplace filter to an image. The sample utilizes the NPP LaplaceBox function, which performs the Laplace operator on an image to highlight areas of rapid intensity change, often used for edge detection in image processing tasks.

This sample is designed to show how to effectively use the NPP library to perform image filtering operations with GPU acceleration, improving performance compared to traditional CPU-based implementations.

### Key Concepts

    CUDA Programming: Leveraging GPU computation with the CUDA toolkit to accelerate image processing tasks.
    NPP Library: A collection of highly optimized functions for image and signal processing, included in the NVIDIA CUDA toolkit.
    Laplace Filter: A second-order derivative operator used in edge detection. It highlights regions of the image where intensity changes abruptly.
    Performance Optimization: Techniques to improve the efficiency and speed of image processing tasks using GPU resources.

### Features

    Application of the Laplace filter using the NPP LaplaceBox function.
    High-performance image processing leveraging CUDA and NPP for GPU acceleration.
    Example implementation with both build and run instructions.

### Requirements

    CUDA Toolkit: The sample requires the NVIDIA CUDA Toolkit to be installed. Ensure that your system has a compatible NVIDIA GPU.
    NPP Library: The NPP library is part of the CUDA Toolkit and provides optimized functions for image processing.
    C++ Compiler: A C++ compiler supporting C++11 or later.

## Build and Run
 
### Linux
    The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:

    $ cd <sample_dir>
    $ make clean build
    $ make run

### Output
The output of the program will display the image with the Laplace filter applied. The filtered image will highlight areas with significant intensity changes (edges).