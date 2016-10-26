#ifndef FASTSPACEDBMETHOD_H
#define FASTSPACEDBMETHOD_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include "optic_flow/OpticFlowCalc.h"
#include <opencv2/gpu/gpu.hpp>

#include "optic_flow/FastSpacedBMOptFlow.h"


class FastSpacedBMethod: public OpticFlowCalc
{
private:
    cv::gpu::GpuMat imCurr_g;
    cv::gpu::GpuMat imPrev_g;

    cv::gpu::GpuMat flowX_g;
    cv::gpu::GpuMat flowY_g;


    int samplePointSize;
    int scanRadius;
    int stepSize;

    double cx,cy,fx,fy;
    double k1,k2,k3,p1,p2;

public:
    FastSpacedBMethod(int i_samplePointSize,
                                         int i_scanRadius,
                                         int i_stepSize,
                                         int i_cx,int i_cy,int i_fx,int i_fy,
                                         int i_k1,int i_k2,int i_k3,int i_p1,int i_p2
                           );

    cv::Point2f processImage(cv::Mat imCurr_t,bool gui,bool debug,
                                     cv::Point midPoint_t);

private:
    void showFlow(const cv::gpu::GpuMat flowx_g, const cv::gpu::GpuMat flowy_g, signed char vXin, signed char vYin);
    void drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion,
                         int step);



};



#endif // FASTSPACEDBMETHOD_H
