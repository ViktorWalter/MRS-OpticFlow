#include "../include/optic_flow/FftMethod.h"

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/imgproc/imgproc.hpp"


FftMethod::FftMethod(int i_frameSize,
                     int i_samplePointSize,
                     int i_numberOfBins)
    {
    frameSize = i_frameSize;
    samplePointSize = i_samplePointSize;
    bins = i_numberOfBins;

    if ((frameSize % 2) == 1){
        frameSize--;
    }
    if((frameSize % samplePointSize) != 0){
        samplePointSize = frameSize;
    }

    xShifts.resize((frameSize/samplePointSize)*(frameSize/samplePointSize));
    yShifts.resize((frameSize/samplePointSize)*(frameSize/samplePointSize));
    bin_arr.resize(bins);

    sqNum = frameSize/samplePointSize;

    first = true;

}

cv::Point2f FftMethod::processImage(cv::Mat imCurr,
                                              bool gui,
                                              bool debug,
                                    cv::Point midPoint){

    if(gui)
        imView = imCurr.clone();


    if(first){
        imCurr.copyTo(imPrev);
        first = false;
    }

    if(debug)
       ROS_INFO("Curr type: %d prev type: %d",imCurr.type(),imPrev.type());

    cv::Mat imCurrF, imPrevF;
    imCurr.convertTo(imCurrF, CV_32FC1);
    imPrev.convertTo(imPrevF, CV_32FC1);

    for(int i = 0;i< sqNum;i++){
        for(int j = 0;j<sqNum;j++){
            xi = i*samplePointSize;
            yi = j*samplePointSize;
            shift = cv::phaseCorrelate(imCurrF(cv::Rect(xi,yi,samplePointSize,samplePointSize)),
                                       imPrevF(cv::Rect(xi,yi,samplePointSize,samplePointSize))
                                       );
            if(abs(shift.x) > samplePointSize/2){
                xShifts.at(i*sqNum + j) = nan("");
                ROS_WARN("FFT - invalid correlation X in rect %d %d",i,j);
            }else{
                xShifts.at(i*sqNum + j) = shift.x;
            }

            if(abs(shift.y) > samplePointSize/2){
                yShifts.at(i*sqNum + j) = nan("");
                ROS_WARN("FFT - invalid correlation Y in rect %d %d",i,j);
            }else{
                yShifts.at(i*sqNum + j) = shift.y;
            }

            if(gui){
                cv::line(imView,
                     cv::Point2i(xi+samplePointSize/2,yi+samplePointSize/2),
                     cv::Point2i(xi+samplePointSize/2,yi+samplePointSize/2)+cv::Point2i((int)(shift.x*5.0),(int)(shift.y*5.0)),
                     cv::Scalar(255));
            }
        }
    }

    xout = weightedMean(&xShifts,-samplePointSize,samplePointSize);
    yout = weightedMean(&yShifts,-samplePointSize,samplePointSize);

    if(debug)
        ROS_INFO("x = %f; y = %f\n",xout,yout);


    imPrev = imCurr.clone();

    if (gui)
    {
        cv::Point2i midPoint = cv::Point2i((imView.size().width/2),(imView.size().height/2));

        cv::line(imView,
                 midPoint,
                 midPoint+cv::Point2i(xout,yout)*6,
                 cv::Scalar(255),2);

        cv::imshow("main",imView);

        cv::waitKey(10);


    }



    return cv::Point2f(xout,yout);
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
    /*
    double step = (max-min)/((double)bins);

    std::fill(bin_arr.begin(),bin_arr.end(),0);
    std:sort(ar->begin(),ar->end());

    // count how many numbers fall into each bin
    double curr_bound = min+step;
    int j = 0;
    for(int i = 0;i<bins;i++){

        if(j >= ar->size()){
            break;
        }

        while(ar->at(j) < curr_bound){

            bin_arr[i]++;
            j++;

            if(j >= ar->size()){
                break;
            }
        }
        curr_bound += step;
    }


    // print output
    /*curr_bound = min+step;
    for(int i = 0;i<bins;i++){
        ROS_INFO("Bin from %f to %f : %d nums",curr_bound-step,curr_bound,bin_arr[i]);
        curr_bound += step;

    }*/
/*
    j = 0;
    int cbi = 0; // current bin starting index
    double loc_sum = 0;
    double sum = 0;

    for(int i = 0;i<bins;i++){
        loc_sum = 0;
        for(j = cbi;j<cbi+bin_arr.at(i);j++){
            loc_sum += ar->at(j);
        }
        sum += loc_sum * ((double) bin_arr.at(i));

        cbi += bin_arr.at(i);

    }

    return sum/((double)ar->size());
  */
}
