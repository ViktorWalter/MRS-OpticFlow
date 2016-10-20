#ifndef FARNMETHOD_H
#define FARNMETHOD_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include "optic_flow/OpticFlowCalc.h"
#include <opencv2/gpu/gpu.hpp>


class FarnMethod: public OpticFlowCalc
{
private:
    cv::gpu::GpuMat imCurr_g;
    cv::gpu::GpuMat imPrev_g;

    cv::gpu::GpuMat flowX_g;
    cv::gpu::GpuMat flowY_g;

    cv::Mat imCurr;
    cv::Mat imPrev;
    cv::Mat imView;
    cv::Point2i midPoint;

    int samplePointSize;
    int scanRadius;

public:
    FarnMethod(int i_samplePointSize,
                int i_numberOfBins
                );

    virtual cv::Point2f processImage(cv::Mat imCurr_t,bool gui,bool debug);

private:
    void showFlow(const char* name, const cv::gpu::GpuMat& d_flow_x, const cv::gpu::GpuMat& d_flow_y);
    void drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion,
                         int step);

};



#endif // FARNMETHOD_H
