#ifndef FFTMETHOD_H
#define FFTMETHOD_H

//#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <image_transport/image_transport.h>
#include "optic_flow/OpticFlowCalc.h"


class FftMethod: public OpticFlowCalc
{
private:

    int frameSize;
    int samplePointSize;

    int imCenterX, imCenterY;    //center of original image
    int xi, yi; //frame corner coordinates

    std::vector<cv::Point2f> speeds;

    int sqNum;

    int numOfChosen,numOfIterations;
    float thresholdRadius_sq;

    cv::Point2d shift;
    cv::Point2f out;

    double xout,yout;

    bool first;
    bool allsac_on;

public:
    FftMethod(int i_frameSize,
              int i_samplePointSize,
              double max_px_speed_t,
              int RansacNumOfChosen,
              int RansacNumOfIter,
              float RansacThresholdRad
              , bool allSac);

    virtual cv::Point2f processImage(cv::Mat imCurr,bool gui,bool debug,cv::Point midPoint);

private:
    double weightedMean(std::vector<double>* ar,double min, double max);
    cv::Point2f ransacMean(std::vector<cv::Point2f> pts, int numOfChosen, float thresholdRadius_sq, int numOfIterations);
    cv::Point2f pointMean(std::vector<cv::Point2f> pts);
    float getDistSq(cv::Point2f p1,cv::Point2f p2);

    cv::Point2f twoPointMean(cv::Point2f p1, cv::Point2f p2);

    cv::Point2f allsacMean(std::vector<cv::Point2f> pts, float thresholdRadius_sq);
};


#endif // FFTMETHOD_H
