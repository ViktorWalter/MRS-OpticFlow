#ifndef BROXMETHOD_H
#define BROXMETHOD_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <image_transport/image_transport.h>
#include "optic_flow/OpticFlowCalc.h"

class BroxMethod: public OpticFlowCalc
{
private:
    cv::gpu::GpuMat imPrev_gf;
    cv::gpu::GpuMat imCurr_gf;

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
    BroxMethod(int samplePointSize_t,int scanRadius_t);

    virtual cv::Point2f processImage(cv::Mat imCurr_t,bool gui,bool debug);
private:
    void showFlow(const char* name, const cv::gpu::GpuMat& d_flow_x, const cv::gpu::GpuMat& d_flow_y);
    void drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion, int step);
};

#endif // BROXMETHOD_H
