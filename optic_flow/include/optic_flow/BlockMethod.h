#ifndef BLOCKMETHOD_H
#define BLOCKMETHOD_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "optic_flow/OpticFlowCalc.h"

class BlockMethod: public OpticFlowCalc
{    
    private:
    cv::Mat imOrigScaled;
    //cv::Mat imCurr;
    cv::Mat imDiff;
    cv::Mat imPrev;

    cv::Mat imView;
    cv::Mat absDiffsMat;
    cv::Mat absDiffsMatSubpix;

    cv::Point2i midPoint;

    int ScaleFactor;

    int samplePointSize;

    int frameSize;
    int maxSamplesSide;

    int scanRadius;
    int scanDiameter;
    int scanCount;
    int stepSize;

    double currentRange;

    int *xHist;
    int *yHist;

    cv::Point2f Refine(cv::Mat imCurr, cv::Mat imPrev, cv::Point2i fullpixFlow,int passes);

    public:
    BlockMethod(int frameSize,
                int samplePointSize,
                int scanRadius,
                int scanDiameter,
                int scanCount,
                int stepSize
                );

    cv::Point2f processImage(cv::Mat imCurr,bool gui,bool debug);






};

#endif // BLOCKMETHOD_H
