#include "../include/optic_flow/FftMethod.h"


FftMethod::FftMethod(int i_frameSize,
                     int i_samplePointSize,
                     double max_px_speed_t)
    {
    frameSize = i_frameSize;
    samplePointSize = i_samplePointSize;
    max_px_speed_sq = pow(max_px_speed_t,2);

    if ((frameSize % 2) == 1){
        frameSize--;
    }
    if((frameSize % samplePointSize) != 0){
        samplePointSize = frameSize;
        ROS_WARN("Oh, what kind of setting for OpticFlow is this? Frame size must be a multiple of SamplePointSize! Forcing FrameSize = SamplePointSize (i.e. one window)..");
    }


    sqNum = frameSize/samplePointSize;

    first = true;





}

std::vector<cv::Point2f> FftMethod::processImage(cv::Mat imCurr,
                                              bool gui,
                                              bool debug,
                                    cv::Point midPoint){

    // save image for GUI
    if(gui)
        imView = imCurr.clone();

    // copy first to second
    if(first){
        imCurr.copyTo(imPrev);
        first = false;
    }

    if(debug)
       ROS_INFO("Curr type: %d prev type: %d",imCurr.type(),imPrev.type());

    // convert images to float images
    cv::Mat imCurrF, imPrevF;
    imCurr.convertTo(imCurrF, CV_32FC1);
    imPrev.convertTo(imPrevF, CV_32FC1);

    // clear the vector with speeds
    speeds.clear();

    // calculate correlation for each window and store it if it doesn't exceed the limit
    for(int i = 0;i< sqNum;i++){
        for(int j = 0;j<sqNum;j++){
            xi = i*samplePointSize;
            yi = j*samplePointSize;
            shift = cv::phaseCorrelate(imCurrF(cv::Rect(xi,yi,samplePointSize,samplePointSize)),
                                       imPrevF(cv::Rect(xi,yi,samplePointSize,samplePointSize))
                                       );

            if(pow(shift.x,2)+pow(shift.y,2) > max_px_speed_sq){
                ROS_WARN("FFT - invalid correlation in window x %d y %d",i,j);
            }else{
                speeds.push_back(cv::Point2f(shift.x,shift.y));
            }

            // draw nice lines if gui is enabled
            if(gui){
                cv::line(imView,
                     cv::Point2i(xi+samplePointSize/2,yi+samplePointSize/2),
                     cv::Point2i(xi+samplePointSize/2,yi+samplePointSize/2)+cv::Point2i((int)(shift.x*5.0),(int)(shift.y*5.0)),
                     cv::Scalar(255));
            }
        }
    }

    // ransac...?

    /*if(allsac_on){
        out = allsacMean(speeds,thresholdRadius_sq);
    }else{
        out = ransacMean(speeds,numOfChosen,thresholdRadius_sq,numOfIterations);
    }*/

    imPrev = imCurr.clone();

    // draw nice center line
    if (gui)
    {
        cv::imshow("main",imView);

        cv::waitKey(10);


    }

    return speeds;
}



double FftMethod::weightedMean(std::vector<double> *ar, double min, double max){
    // sort numbers from array into bins and then preform weighted mean based on the number of numbers in each bin
    double sum = 0;
    int size = 0;
    for(int i = 0;i < ar->size();i++){
        if(!isnan(ar->at(i))){
            sum += ar->at(i);
            size += 1;
        }
    }
    if(size == 0)
        return nan("");
    else
        return sum/((double)size);
}

cv::Point2f FftMethod::ransacMean(std::vector<cv::Point2f> pts, int numOfChosen, float thresholdRadius_sq, int numOfIterations){
    if(pts.size() <= numOfChosen){   // weve got less or same number (or zero?) of points as number to choose
        return pointMean(pts);
    }

    cv::Point2f bestIter; // save the best mean here
    uint bestIter_num = 0; // save number of points in best mean
    std::vector<cv::Point2f> currIter; // here goes current iteration
    cv::Point2f currMean;

    for(uint i=0;i < numOfIterations;i++){ // ITERATE!!!
        currIter.clear();

        for(uint j=0;j<numOfChosen;j++){ // choose some points (one point can be chosen more times...)
            currIter.push_back(pts[rand()%pts.size()]);
        }

        currMean = pointMean(currIter); // get the mean

        currIter.clear(); // clear this array

        for(uint k=0;k<pts.size();k++){ // choose those in threshold
            if(getDistSq(currMean,pts[k]) < thresholdRadius_sq){
                currIter.push_back(pts[k]);
            }
        }

        if(currIter.size() > bestIter_num){
            bestIter_num = currIter.size();
            bestIter = pointMean(currIter);
        }
    }

    return bestIter;
}


cv::Point2f FftMethod::pointMean(std::vector<cv::Point2f> pts){
    float mx = 0;
    float my = 0;
    int numPts = 0;
    for(uint i = 0;i < pts.size();i++){
        if(!isnan(pts[i].x) && !isnan(pts[i].y)){
            mx += pts[i].x;
            my += pts[i].y;
            numPts++;
        }
    }

    if(numPts > 0){
        // now we're talking..
        return cv::Point2f(mx/(float)numPts,my/(float)numPts);
    }

    // what do you want me to do with this?
    return cv::Point2f(nanf(""),nanf(""));
}

float FftMethod::getDistSq(cv::Point2f p1,cv::Point2f p2){
    return pow(p1.x - p2.x,2) + pow(p1.y - p2.y,2);
}

cv::Point2f FftMethod::twoPointMean(cv::Point2f p1, cv::Point2f p2){
    return cv::Point2f((p1.x+p2.x)/2, (p1.y+p2.y)/2);
}

cv::Point2f FftMethod::allsacMean(std::vector<cv::Point2f> pts, float thresholdRadius_sq){
    // For every two points get their mean and do the typical RANSAC things...
    if(pts.size() <= 2){   // weve got less or same number (or zero?) of points as number to choose
        return pointMean(pts);
    }

    cv::Point2f currMean;
    std::vector<cv::Point2f> currIter;

    int bestIter_num = 0;
    cv::Point2f bestIter;

    for(uint i = 0;i < pts.size();i++){
        for(uint j=i;j<pts.size();j++){ // iterate over all pairs

            currIter.clear();

            currMean = twoPointMean(pts[i],pts[j]); // calc mean

            for(uint k=0;k<pts.size();k++){ // choose those in threshold
                if(getDistSq(currMean,pts[k]) < thresholdRadius_sq){
                    currIter.push_back(pts[k]);
                }
            }

            if(currIter.size() > bestIter_num){
                bestIter_num = currIter.size();
                bestIter = pointMean(currIter);
                if(bestIter_num >= pts.size()){
                    return bestIter;
                }
            }
        }
    }

    return bestIter;
}
