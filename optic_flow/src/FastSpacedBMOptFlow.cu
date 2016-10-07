#include "optic_flow/FastSpacedBMOptFlow.h"

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/gpu/gpumat.hpp>

#define arraySize 20

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

__global__ void _FastSpacedBMOptFlow_kernel(const cv::gpu::PtrStepSzb input_1,
                                            const cv::gpu::PtrStepSzb input_2,
                                            cv::gpu::PtrStepSz<signed char> output_X,
                                            cv::gpu::PtrStepSz<signed char> output_Y,
                                            int blockSize,
                                            int blockStep,
                                            int scanRadius,
                                            int width,
                                            int height)
{
    int scanDiameter = scanRadius*2+1;
    __shared__ int abssum[arraySize][arraySize];


    if( ((blockIdx.x+1) * (blockSize + 1+scanRadius*2) < width)
            &&
            ((blockIdx.y+1) * (blockSize + 1+scanRadius*2)) < height )
    {

            for (int i=0;i<blockSize;i++)
            {
                for (int j=0;j<blockSize;j++)
                {
                    abssum[threadIdx.y][threadIdx.x]+=
                            (abs(input_1((blockIdx.y + scanRadius + i),(blockIdx.x + scanRadius + j))
                            -
                            input_2((blockIdx.y + i + threadIdx.y),
                                    (blockIdx.x + j + threadIdx.x))));
                }

            }

            __syncthreads();
            __shared__ int minval[arraySize];
            __shared__ signed char minX[arraySize];
            signed char minY;

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
                output_Y(blockIdx.y,blockIdx.x) = minY;
                output_X(blockIdx.y,blockIdx.x) = minX[minY+scanRadius];
            }



    }





}

__global__ void _HistogramMaximum(const cv::gpu::PtrStepSz<signed char> input,
                                  int scanRadius,
                                  signed char *value)
{

    __shared__ int Histogram[arraySize];

    if ((threadIdx.x < arraySize) && (threadIdx.y == 0))
        Histogram[threadIdx.x] = 0;


    __syncthreads();
    Histogram[input(threadIdx.y,threadIdx.x)+scanRadius]++;
    __syncthreads();


    if ( (threadIdx.y == 0) && (threadIdx.x == 0))
    {
        int MaxIndex = 0;
        char  MaxVal = 0;
        for (int i=0;i<blockDim.y;i++)
        {
            if (MaxVal < Histogram[i])
            {
                MaxVal = Histogram[i];
                MaxIndex = i;
            }
        }
        *value = MaxIndex - scanRadius;
    }

}

__global__ void _CopyMatrix(const cv::gpu::PtrStepSzb input,
                            cv::gpu::PtrStepb output,
                            int blockSize,
                            int width,
                            int height) //test for CUDA CV basics
{
        output(threadIdx.y,threadIdx.x) = input(threadIdx.y,threadIdx.x);
}

void ResetCudaDevice()
{

    SAFE_CALL(cudaDeviceReset(),"Killing previous kernels Failed!");
}

void FastSpacedBMOptFlow(cv::InputArray _imPrev, cv::InputArray _imCurr,
                         cv::OutputArray _imOutX, cv::OutputArray _imOutY,
                         int blockSize,
                         int blockStep,
                         int scanRadius,
                         signed char &outX,
                         signed char &outY)
{
    const cv::gpu::GpuMat imPrev = _imPrev.getGpuMat();
    const cv::gpu::GpuMat imCurr = _imCurr.getGpuMat();
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

    const dim3 block(scanDiameter, scanDiameter);
    const dim3 grid((imPrev.cols)/blockszX, (imPrev.rows)/blockszY);

    _imOutX.create(grid.x,grid.y,CV_8UC1);
    const cv::gpu::GpuMat imOutX = _imOutX.getGpuMat();
    _imOutY.create(grid.x,grid.y,CV_8UC1);
    const cv::gpu::GpuMat imOutY = _imOutY.getGpuMat();

    //_CopyMatrix<<<1,blockM>>>(imPrev,imOutX,blockSize,imPrev.cols,imPrev.rows);
    _FastSpacedBMOptFlow_kernel<<<grid,block>>>(imPrev,imCurr,imOutX,imOutY,
                                                blockSize,blockStep,scanRadius,
                                                imCurr.cols, imCurr.rows);

    signed char outX_l;
    signed char outY_l;

    signed char* outX_g;
    signed char* outY_g;
    cudaMalloc(&outX_g, sizeof(signed char));
    cudaMalloc(&outY_g, sizeof(signed char));

    _HistogramMaximum<<<1,grid>>>(imOutX,scanRadius,outX_g);
    _HistogramMaximum<<<1,grid>>>(imOutY,scanRadius,outY_g);

    SAFE_CALL(cudaMemcpy(&outX_l,outX_g,sizeof(signed char),cudaMemcpyDeviceToHost),"Memcpy to host failed");
    SAFE_CALL(cudaMemcpy(&outY_l,outY_g,sizeof(signed char),cudaMemcpyDeviceToHost),"Memcpy to host failed");

    cudaFree(outX_g);
    cudaFree(outY_g);

   SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

   outX = outX_l;
   outY = outY_l;



}
