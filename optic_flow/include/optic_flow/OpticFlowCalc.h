#ifndef OPTICFLOWCALC_H
#define OPTICFLOWCALC_H

#include <cv_bridge/cv_bridge.h>

class OpticFlowCalc
{
    public:
        virtual cv::Point2f processImage(cv::Mat imCurr,
                                                 bool gui,
                                                 bool debug);
};

#endif // OPTICFLOWCALC_H
