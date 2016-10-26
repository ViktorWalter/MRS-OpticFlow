#ifndef OPTICFLOWCALC_H
#define OPTICFLOWCALC_H

#include <cv_bridge/cv_bridge.h>

class OpticFlowCalc
{

    public:
           virtual cv::Point2f processImage(cv::Mat imCurr,
                                         bool gui,
                                         bool debug,
                                         cv::Point midPoint_t)
    {
            midPoint = midPoint_t;
    }

            void setImPrev(cv::Mat imPrev_t)
            {
                imPrev = imPrev_t;

            }

protected:
     cv::Mat imPrev, imCurr, imView;
     cv::Point2i midPoint;


};

#endif // OPTICFLOWCALC_H
