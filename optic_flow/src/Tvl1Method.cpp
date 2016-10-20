#include "../include/optic_flow/Tvl1Method.h"



Tvl1Method::Tvl1Method(int i_samplePointSize,
                       int i_numberOfBins
                       )
    {
    samplePointSize = i_samplePointSize;
    scanRadius = i_numberOfBins;
}

cv::Point2f Tvl1Method::processImage(cv::Mat imCurr_t,
                                              bool gui,
                                              bool debug){



    imCurr = imCurr_t;
    imPrev_g.upload(imPrev);
    imCurr_g.upload(imCurr);

    clock_t beginGPU = clock();
    fastBM( imCurr_g,imPrev_g, flowX_g,flowY_g,8);
    clock_t end = clock();
    double elapsed_secs = double(end - beginGPU) / CLOCKS_PER_SEC;

    if(debug)
        ROS_INFO("Tvl1: %f msec",elapsed_secs*1000);

    if (gui)
        showFlow("TVL1", flowX_g, flowY_g);


    cv::Scalar outputX;
    cv::Scalar outputY;
    outputX = cv::gpu::sum(flowX_g,buffer_g)/(double)flowX_g.size().area();
    outputY = cv::gpu::sum(flowY_g,buffer_g)/(double)flowY_g.size().area();




    imPrev = imCurr.clone();
    return cv::Point2f((float)outputX.val[0],(float)outputY.val[0]);
}

void Tvl1Method::showFlow(const char* name, const cv::gpu::GpuMat& d_flow_x, const cv::gpu::GpuMat& d_flow_y)
{
    cv::Mat flowx(d_flow_x);
    cv::Mat flowy(d_flow_y);

    cv::Mat out;
    drawOpticalFlow(flowx, flowy, out, 10, 0);

    //int vx = getHistMaxGPU(d_flow_x);
    //int vy = getHistMaxGPU(d_flow_y);
    double vx, vy;

    vx = cv::mean(flowx)[0];
    vy = cv::mean(flowy)[0];
    ROS_INFO("vx = %f; vy=%f",vx,vy);

    cv::line(imView,
              midPoint,
              midPoint+cv::Point2i((int)(vx*4),(int)(vy*4)),
              cv::Scalar(255),2);

    cv::imshow("Main", imView);
    cv::waitKey(10);
}

void Tvl1Method::drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion,
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
