#ifndef UTILITYFUNCTIONS_H
#define UTILITYFUNCTIONS_H
#include <math.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

void rotate2d(double &x, double &y, double alpha);
void rotate2d(cv::Point2f &pt, double alpha);
cv::Point2f pointMean(std::vector<cv::Point2f> pts);
float getDistSq(cv::Point2f p1,cv::Point2f p2);
float getNormSq(cv::Point2f p1);
cv::Point2f twoPointMean(cv::Point2f p1, cv::Point2f p2);
cv::Point2f allsacMean(std::vector<cv::Point2f> pts, float thresholdRadius_sq);
void multiplyAllPts(std::vector<cv::Point2f> &v, float mulx, float muly);
void rotateAllPts(std::vector<cv::Point2f> &v,double alpha);
void addToAll(std::vector<cv::Point2f> &v,float adx, float ady);
cv::Point2f ransacMean(std::vector<cv::Point2f> pts, int numOfChosen, float thresholdRadius_sq, int numOfIterations);
std::vector<cv::Point2f> getOnlyInAbsBound(std::vector<cv::Point2f> v,float low,float up);
std::vector<cv::Point2f> removeNanPoints(std::vector<cv::Point2f> v);
#endif // UTILITYFUNCTIONS_H
