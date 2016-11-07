#ifdef CUDA_SUPPORTED
#include "../include/optic_flow/FastSpacedBMethod.h"


FastSpacedBMethod::FastSpacedBMethod(int i_samplePointSize,
                                     int i_scanRadius,
                                     int i_stepSize,
                                     int i_cx,int i_cy,int i_fx,int i_fy,
                                     int i_k1,int i_k2,int i_k3,int i_p1,int i_p2)
    {

    samplePointSize = i_samplePointSize;
    scanRadius = i_scanRadius;
    stepSize = i_stepSize;
    cx = i_cx;
    cy = i_cy;
    fx = i_fx;
    fy = i_fy;
    k1 = i_k1;
    k2 = i_k2;
    k3 = i_k3;
    p1 = i_p1;
    p2 = i_p2;
}

cv::Point2f FastSpacedBMethod::processImage(cv::Mat imCurr_t,
                                              bool gui,
                                              bool debug,
                                             cv::Point midPoint_t)
{
    midPoint = midPoint_t;

    signed char outputX;
    signed char outputY;

    imCurr = imCurr_t;

    imPrev_g.upload(imPrev);
    imCurr_g.upload(imCurr);


    FastSpacedBMOptFlow(imCurr_g,imPrev_g, flowX_g,flowY_g,samplePointSize,stepSize,scanRadius,
                        cx, cy, fx,fy, k1, k2, k3, p1, p2,
                        outputX,
                        outputY
                        );

    if (debug)
    {
        ROS_INFO("out: %dx%d",flowX_g.cols,flowX_g.rows);
    }
    if (gui)
    {
        showFlow(flowX_g, flowY_g, outputX, outputY);
    }

    imPrev = imCurr.clone();

    return cv::Point2f(outputX,outputY);

}

void FastSpacedBMethod::showFlow(const cv::gpu::GpuMat flowx_g, const cv::gpu::GpuMat flowy_g, signed char vXin, signed char vYin)
{
    cv::Mat flowx = cv::Mat(flowx_g);
    cv::Mat flowy = cv::Mat(flowy_g);

    cv::Mat out;
    drawOpticalFlow(flowx, flowy, out, 10, stepSize);

    cv::line(imView,
              midPoint,
              midPoint+cv::Point2i(((int)vXin*4),((int)vYin*4)),
              cv::Scalar(255),2);

    cv::imshow("Main", imView);
    cv::waitKey(10);
}

void FastSpacedBMethod::drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion,
                     int step)
{
    imView = imCurr.clone();

    for (int y = 0; y < flowx.rows; y++)
    {
        for (int x = 0; x < flowx.cols; x++)
        {
            if ((abs(flowx(y, x)) > scanRadius) || (abs(flowy(y, x))> scanRadius))
            {
                ROS_WARN("Flow out of bounds: X:%d, Y:%d",flowx(y, x),flowy(y, x));
                //continue;
            }
            cv::Point2i startPos(x*(step+samplePointSize)+(samplePointSize/2+scanRadius),
                                 y*(step+samplePointSize)+(samplePointSize/2+scanRadius));

            cv::Point2i u(flowx(y, x), flowy(y, x));
            cv::line(imView,
                 startPos,
                 startPos+u,
                 cv::Scalar(255));

        }
    }
    dst = imView;
}
#endif
