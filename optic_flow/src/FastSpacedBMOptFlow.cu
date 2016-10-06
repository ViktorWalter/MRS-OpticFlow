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
    int scanDiameter = scanRadius*2+1;
    __shared__ int abssum[20][20];

    if( (blockIdx.x * (blockSize + 1+scanRadius*2) < width) && (blockIdx.y * (blockSize + 1+scanRadius*2)) < height )
    {
            for (int i=0;i<blockSize;i++)
            {
                for (int j=0;j<blockSize;i++)
                {
                    abssum[threadIdx.y][threadIdx.x]+=
                            (abs(input_1[
                            (width * (blockIdx.y + i))    //y
                            + blockIdx.x + scanRadius + j] -        //x
                            input_2[
                            (width * (blockIdx.y + i + threadIdx.y - (blockSize/2))) //y
                            + blockIdx.x + j + threadIdx.x - (blockSize/2)]));      //x
                }

            }

            __syncthreads();
            __shared__ int minval[20];
            char minX[20];
            char minY;

            if (threadIdx.y == 0)
            {
                minval[threadIdx.x] = abssum[threadIdx.x][0];
                minX[threadIdx.x] = -scanRadius;
                for (int i=1;i<scanDiameter;i++)
                {
                    if (minval[threadIdx.x] > abssum[threadIdx.x][i])
                    {
                        minval[threadIdx.x] = abssum[threadIdx.x][i];
                        minX[threadIdx.x] = threadIdx.x-scanRadius;
                    }
                }
            }
             __syncthreads();

            if ( (threadIdx.y == 0) && (threadIdx.x == 0))
            {
                int minvalFin = minval[0];
                minY = -scanRadius;
                for (int i=1;i<scanDiameter;i++)
                {
                    if (minvalFin > minval[i])
                    {
                        minvalFin = minval[i];
                        minY = i-scanRadius;
                    }
                }
                output_Y[width*blockIdx.y + blockIdx.x] = minY;
                output_X[width*blockIdx.y + blockIdx.x] = minX[minY+scanRadius];
            }



    }





}

void FastSpacedBMOptFlow(cv::gpu::GpuMat &imPrev, cv::gpu::GpuMat &imCurr,
                         cv::gpu::GpuMat &imOutX, cv::gpu::GpuMat &imOutY,
                         int blockSize,
                         int blockStep,
                         int scanRadius)
{
    if (imPrev.size() != imCurr.size())
    {
        std::fprintf(stderr,"Input images do not match in sizes!");
        std::cin.get();
        exit(EXIT_FAILURE);
    }
    if ((imPrev.type() != CV_8UC1) || (imCurr.type() != CV_8UC1))
    {
        std::fprintf(stderr,"Input image/s are not of the CV_8UC1 type!");
        std::cin.get();
        exit(EXIT_FAILURE);
    }

    int scanDiameter = 2*scanRadius+1;
    int blockszX = scanDiameter+blockSize;
    int blockszY = scanDiameter+blockSize;

    imOutX = cv::gpu::GpuMat((imPrev.cols)/blockszX, (imPrev.rows)/blockszY,
                             CV_8SC1);
    imOutY = cv::gpu::GpuMat((imPrev.cols)/blockszX, (imPrev.rows)/blockszY,
                             CV_8SC1);

    unsigned char* pi1 = (unsigned char*)imPrev.data;
    unsigned char* pi2 = (unsigned char*)imCurr.data;
    unsigned char* po1 = (unsigned char*)imOutX.data;
    unsigned char* po2 = (unsigned char*)imOutY.data;


    const dim3 block(scanDiameter, scanDiameter);
    const dim3 grid((imPrev.cols)/blockszX, (imPrev.rows)/blockszY);

    _FastSpacedBMOptFlow_kernel<<<grid,block>>>(pi1,pi2,po1,po2,
                                                blockSize,blockStep,scanRadius,
                                                imCurr.cols, imCurr.rows);




   SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");


}
