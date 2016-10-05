#include "optic_flow/FastSpacedBMOptFlow.h"

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/gpu/gpumat.hpp>

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        std::fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",
                msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void _FastSpacedBMOptFlow_kernel(unsigned char* input_1,
                                    unsigned char* input_2,
                                    unsigned char* output_X,
                                    unsigned char* output_Y,
                                    int blockSize,
                                    int blockStep,
                                    int scanRadius,
                                    int width,
                                    int height)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex<width) && (yIndex<height))
    {
        if
        {

        }



    }
}

void FastSpacedBMOptFlow(cv::gpu::GpuMat &imPrev, cv::gpu::GpuMat &imCurr,
                         cv::gpu::GpuMat &imOutX, cv::gpu::GpuMat &imOutY,
                         int blockSize,
                         int blockStep,
                         int scanRadius)
{
    unsigned char* pi1 = (unsigned char*)imPrev.data;
    unsigned char* pi2 = (unsigned char*)imCurr.data;
    unsigned char* po1 = (unsigned char*)imOutX.data;
    unsigned char* po2 = (unsigned char*)imOutY.data;
    const dim3 block(blockSize, blockSize);
    const dim3 grid((imPrev.cols + block.x - 1)/block.x, (imPrev.rows + block.y - 1)/block.y);

    _FastSpacedBMOptFlow_kernel<<<grid,block>>>(pi1,pi2,po1,po2,
                                                blockSize,blockStep,scanRadius,
                                                imCurr.cols, imCurr.rows);




   SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");


}
