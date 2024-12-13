/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

// Prints out NPP library and CUDA version information
bool printNPPInfo(int argc, char *argv[]) {
    const NppLibraryVersion *libVer = nppGetLibVersion();
    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    return checkCudaCapabilities(1, 0);
}

int main(int argc, char *argv[]) {
    printf("%s Starting...\n\n", argv[0]);

    try {
        std::string inputFilename;
        char *filePath;

        // Initialize CUDA device
        findCudaDevice(argc, (const char **)argv);

        // Print NPP and CUDA info
        if (!printNPPInfo(argc, argv)) {
            exit(EXIT_SUCCESS);
        }

        // Check for input file path argument, else set default filename
        if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        } else {
            filePath = sdkFindFilePath("aditya.pgm", argv[0]);  // Updated file name
        }

        inputFilename = filePath ? filePath : "aditya.pgm"; // Set filename

        // Check if the input file exists
        std::ifstream infile(inputFilename, std::ifstream::in);
        if (infile.good()) {
            std::cout << "Successfully opened: <" << inputFilename << ">" << std::endl;
            infile.close();
        } else {
            std::cerr << "Unable to open: <" << inputFilename << ">" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Prepare output filename by modifying input file name
        std::string resultFilename = inputFilename;
        size_t dotPos = resultFilename.rfind('.');
        if (dotPos != std::string::npos) {
            resultFilename = resultFilename.substr(0, dotPos);  // Remove file extension
        }
        resultFilename += "_filterLaplaceBorder.pgm";  // Add filter suffix

        // Allow user to specify output filename via command line
        if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
            resultFilename = outputFilePath;
        }

        // Declare and load host image (8-bit grayscale)
        npp::ImageCPU_8u_C1 hostSrc;
        npp::loadImage(inputFilename, hostSrc);  // Load image into host memory

        // Upload image to device memory
        npp::ImageNPP_8u_C1 deviceSrc(hostSrc);

        // Define filter size (5x5 box filter)
        NppiSize maskSize = {5, 5};
        NppiMaskSize maskEnum(NPP_MASK_SIZE_5_X_5);

        // Define source image size and ROI (Region of Interest)
        NppiSize srcSize = {deviceSrc.width(), deviceSrc.height()};
        NppiPoint srcOffset = {0, 0};  // No offset
        NppiSize roiSize = {deviceSrc.width(), deviceSrc.height()};

        // Allocate device memory for the filtered output image
        npp::ImageNPP_8u_C1 deviceDst(roiSize.width, roiSize.height);

        // Define anchor point for filter (center of the mask)
        NppiPoint anchor = {maskSize.width / 2, maskSize.height / 2};

        // Apply Laplace filter with border replicate handling
        NPP_CHECK_NPP(nppiFilterLaplaceBorder_8u_C1R(
            deviceSrc.data(), deviceSrc.pitch(), srcSize, srcOffset,
            deviceDst.data(), deviceDst.pitch(), roiSize, maskEnum,
            NPP_BORDER_REPLICATE));

        // Copy filtered image back to host memory
        npp::ImageCPU_8u_C1 hostDst(deviceDst.size());
        deviceDst.copyTo(hostDst.data(), hostDst.pitch());

        // Save the filtered image to disk
        saveImage(resultFilename, hostDst);
        std::cout << "Saved image: " << resultFilename << std::endl;

        // Free device memory
        nppiFree(deviceSrc.data());
        nppiFree(deviceDst.data());

        exit(EXIT_SUCCESS);
    } catch (npp::Exception &exception) {
        std::cerr << "Error occurred: " << exception << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cerr << "An unknown error occurred. Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}