
#include "../include/optic_flow/utilityFunctions.h"

void rotate2d(double &x, double &y, double alpha)
{
    double cs = cos(alpha);// cos and sin of rot maxtrix
    double sn = sin(alpha);

    double x_n = x*cs-y*sn;
    double y_n = x*sn+y*cs;
    x = x_n;
    y = y_n;
}

void rotate2d(cv::Point2f &pt, double alpha)
{
    double cs = cos(alpha);// cos and sin of rot maxtrix
    double sn = sin(alpha);

    double x_n = pt.x*cs-pt.y*sn;
    double y_n = pt.x*sn+pt.y*cs;
    pt.x = x_n;
    pt.y = y_n;
}


cv::Point2f pointMean(std::vector<cv::Point2f> pts){
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

float getDistSq(cv::Point2f p1,cv::Point2f p2){
    return pow(p1.x - p2.x,2) + pow(p1.y - p2.y,2);
}

float getNormSq(cv::Point2f p1){
    return pow(p1.x,2) + pow(p1.y,2);
}


cv::Point2f twoPointMean(cv::Point2f p1, cv::Point2f p2){
    return cv::Point2f((p1.x+p2.x)/2, (p1.y+p2.y)/2);
}

cv::Point2f allsacMean(std::vector<cv::Point2f> pts, float thresholdRadius_sq){
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

void multiplyAllPts(std::vector<cv::Point2f> &v,float mulx,float muly){
    for(uint i=0;i<v.size();i++){
        v[i].x *= mulx;
        v[i].y *= mulx;
    }
}

void rotateAllPts(std::vector<cv::Point2f> &v,double alpha){
    for(uint i=0;i<v.size();i++){
        rotate2d(v[i],alpha);
    }
}

void addToAll(std::vector<cv::Point2f> &v,float adx, float ady){
    for(uint i=0;i<v.size();i++){
        v[i].x += adx;
        v[i].y += ady;
    }
}

cv::Point2f ransacMean(std::vector<cv::Point2f> pts, int numOfChosen, float thresholdRadius_sq, int numOfIterations){
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

std::vector<cv::Point2f> getOnlyInAbsBound(std::vector<cv::Point2f> v,float low,float up){
    std::vector<cv::Point2f> ret;
    float lowSq = pow(low,2);
    float upSq = pow(up,2);
    float n;
    for(int i=0;i<v.size();i++){
        n = getNormSq(v[i]);
        if(n > lowSq && n < upSq){
            ret.push_back(v[i]);
        }
    }
    return ret;

}

std::vector<cv::Point2f> removeNanPoints(std::vector<cv::Point2f> v){
    std::vector<cv::Point2f> ret;
    for(int i=0;i<v.size();i++){
        if(!isnanf(v[i].x) && !isnanf(v[i].y)){
            ret.push_back(v[i]);
        }
    }
    return ret;
}
