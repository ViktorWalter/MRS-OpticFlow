#ifndef FASTBMOF_H_
#define FASTBMOF_H_
#include <opencv2/core.hpp>

void FastSpacedBMOptFlow(cv::gpu::GpuMat &imPrev, cv::gpu::GpuMat &imCurr,
                         cv::gpu::GpuMat &imOutX, cv::gpu::GpuMat &imOutY,
                         int blockSize,
                         int blockStep,
                         int scanRadius);

#endif  // FASTBMOF_H_
