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

    int bins;
    std::vector<int> bin_arr;
    std::vector<double> xShifts;
    std::vector<double> yShifts;
    cv::Point2d shift;

    int sqNum;

    double xout,yout;

    bool first;

public:
    FftMethod(int i_frameSize,
                int i_samplePointSize,
                int i_numberOfBins);

    virtual cv::Point2f processImage(cv::Mat imCurr,bool gui,bool debug,cv::Point midPoint);

private:
    double weightedMean(std::vector<double>* ar,double min, double max);

};

#endif // FFTMETHOD_H
