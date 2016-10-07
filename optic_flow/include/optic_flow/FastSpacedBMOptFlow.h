#ifndef FASTBMOF_H_
#define FASTBMOF_H_
#include <opencv2/core.hpp>

void FastSpacedBMOptFlow(cv::InputArray _imPrev, cv::InputArray _imCurr,
                         cv::OutputArray _imOutX, cv::OutputArray _imOutY,
                         int blockSize,
                         int blockStep,
                         int scanRadius,
                         signed char &outX,
                         signed char &outY);

void ResetCudaDevice();

#endif  // FASTBMOF_H_
