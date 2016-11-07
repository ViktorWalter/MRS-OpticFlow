#ifdef CUDA_SUPPORTED
#include "../include/optic_flow/BroxMethod.h"

BroxMethod::BroxMethod(int samplePointSize_t,int scanRadius_t)
    {
    samplePointSize = samplePointSize_t;
    scanRadius = scanRadius_t;
}

cv::Point2f BroxMethod::processImage(cv::Mat imCurr_t,
                                              bool gui,
                                              bool debug){

    cv::gpu::BroxOpticalFlow brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

    imCurr = imCurr_t;
    imPrev_g.upload(imPrev);
    imCurr_g.upload(imCurr);

    cv::gpu::GpuMat imPrev_gf;
    cv::gpu::GpuMat imCurr_gf;

    imPrev_g.convertTo(imPrev_gf, CV_32F, 1.0 / 255.0);
    imCurr_g.convertTo(imCurr_gf, CV_32F, 1.0 / 255.0);



    clock_t beginGPU = clock();
    brox(imCurr_gf, imPrev_gf, flowX_g,flowY_g);
    clock_t end = clock();
    double elapsed_secs = double(end - beginGPU) / CLOCKS_PER_SEC;

    if(debug)
        ROS_INFO("Brox: %f msec",elapsed_secs*1000);

    //cout << "Brox : " << elapsed_secs*1000 << " msec" << endl;

    if (gui)
        showFlow("Brox", flowX_g,flowY_g);

    imPrev = imCurr.clone();


    return cv::Point2f(0,0);
}

void BroxMethod::showFlow(const char* name, const cv::gpu::GpuMat& d_flow_x, const cv::gpu::GpuMat& d_flow_y)
{
    cv::Mat flowx(d_flow_x);
    cv::Mat flowy(d_flow_y);

    cv::Mat out;
    drawOpticalFlow(flowx, flowy, out, 10., 0);

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

void BroxMethod::drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion,
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
#endif //CUDA_SUPPORTED
