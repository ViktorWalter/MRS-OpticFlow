#define maxTerraRange 8.0

//#define CUDA_SUPPORTED


#include <ros/ros.h>
#include <tf/tf.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include "optic_flow/utilityFunctions.h"
using namespace std;


//#include <opencv2/gpuoptflow.hpp>
//#include <opencv2/gpulegacy.hpp>
//#include <opencv2/gpuimgproc.hpp>
//#include <time.h>



#include "optic_flow/OpticFlowCalc.h"
#include "optic_flow/BlockMethod.h"
#include "optic_flow/FftMethod.h"
#include "optic_flow/FastSpacedBMMethod_OCL.h"

#ifdef CUDA_SUPPORTED
#include "optic_flow/BroxMethod.h"
#include "optic_flow/FarnMethod.h"
#include "optic_flow/FastSpacedBMethod.h"
#include "optic_flow/Tvl1Method.h"
#include "optic_flow/FastSpacedBMOptFlow.h"

#include <opencv2/gpu/gpu.hpp>

#endif



namespace enc = sensor_msgs::image_encodings;

struct PointValue
{
    int value;
    cv::Point2i location;
};

class OpticFlow
{
public:
    OpticFlow(ros::NodeHandle& node)
    {
        coordsAcquired = false;
        first = true;
        checkAccel = false;

        ros::NodeHandle private_node_handle("~");
        private_node_handle.param("DEBUG", DEBUG, bool(false));

        private_node_handle.param("useCuda", useCuda, bool(false));
        private_node_handle.param("method", method, int(0));
        /* methods:
         *      0 - Brox, only CUDA
         *      1 - Farn, only CUDA
         *      2 - BM, only CUDA
         *      3 - BM (CUDA - FastSpaced)
         *      4 - FFT, only CPU
        */

        if(method < 3 && !useCuda){
            ROS_ERROR("Specified method support only CUDA");
        }

        private_node_handle.param("ScanRadius", scanRadius, int(8));
        if ((scanRadius>15) && (useCuda) && (method==3))
        {
            ROS_INFO("This CUDA method only allows scanning size of up to 15 pixels. Trimming to 15 p.");
            scanRadius = 15;
        }

        private_node_handle.param("FrameSize", frameSize, int(64));
        private_node_handle.param("SamplePointSize", samplePointSize, int(8));
        private_node_handle.param("NumberOfBins", numberOfBins, int(20));


        private_node_handle.param("StepSize", stepSize, int(0));

        private_node_handle.param("gui", gui, bool(false));
        private_node_handle.param("publish", publish, bool(true));

        private_node_handle.param("useOdom",useOdom,bool(false));

        std::string ImgPath, RangerPath;
        bool ImgCompressed;
        private_node_handle.param("CameraImagePath", ImgPath, std::string("/uav/mv_25001879/image_raw"));
				ROS_INFO("CamImPath:='%s'",ImgPath.c_str());
        private_node_handle.param("CameraImageCompressed", ImgCompressed, bool(false));

        private_node_handle.param("RangerDataPath", RangerPath, std::string("/uav/terarangerone"));

        private_node_handle.param("ScaleFactor", ScaleFactor, int(1));

        private_node_handle.param("RansacNumOfChosen", RansacNumOfChosen, int(2));
        private_node_handle.param("RansacNumOfIter", RansacNumOfIter, int(5));
        float RansacThresholdRad;
        private_node_handle.param("RansacThresholdRad", RansacThresholdRad, float(4));
        RansacThresholdRadSq = pow(RansacThresholdRad,2);
        private_node_handle.param("Allsac", Allsac, bool(true));


        if(Allsac && RansacNumOfChosen != 2){
            ROS_WARN("When Allsac is enabled, the RansacNumOfChosen can be only 2.");
        }

        ROS_INFO("RANSAC settings - ALLsac: %c, Num of chosen: %d, Num of Iters: %d, threshold radius: %f",Allsac ? 'y' : 'n',RansacNumOfChosen,RansacNumOfIter,RansacThresholdRad);

        if(RAND_MAX < 100 && !Allsac){
            ROS_WARN("Why is RAND_MAX set to only %d? Ransac in OpticFlow won't work properly!",RAND_MAX);
        }

        std::vector<double> camMat;
        private_node_handle.getParam("camera_matrix/data", camMat);
        fx = camMat[0];
        cx = camMat[2];
        fy = camMat[4];
        cy = camMat[5];
        std::vector<double> distCoeffs;
        private_node_handle.getParam("distortion_coefficients/data",distCoeffs);
        k1 = distCoeffs[0];
        k2 = distCoeffs[1];
        k3 = distCoeffs[4];
        p1 = distCoeffs[2];
        p2 = distCoeffs[3];


        private_node_handle.getParam("image_width", expectedWidth);

        if ((frameSize % 2) == 1)
        {
            frameSize--;
        }
        scanDiameter = (2*scanRadius+1);
        scanCount = (scanDiameter*scanDiameter);


        private_node_handle.param("cameraRotated", cameraRotated, bool(true));
        //private_node_handle.getParam("camera_rotation_matrix/data", camRot);
        private_node_handle.getParam("alpha", gamma);

        private_node_handle.getParam("max_px_speed", max_px_speed_t);
        private_node_handle.getParam("max_speed", maxSpeed);
        private_node_handle.getParam("max_acceleration", maxAccel);



        if (useCuda)
        {
            #ifdef CUDA_SUPPORTED
                ResetCudaDevice();
            #else
                ROS_ERROR("Not compiled for CUDA. If you want to use it, #define CUDA_SUPPORTED in optic_flow.cpp");
            #endif
        }


        if(useCuda && method != 3){
            ROS_WARN("Method does not have cuda/noncuda version.");
        }



        switch(method){
            #ifdef CUDA_SUPPORTED
            case 0:
            {
                processClass = new BroxMethod(samplePointSize,scanRadius);
                break;
            }
            case 1:
            {
                processClass = new FarnMethod(samplePointSize,scanRadius);
                break;
            }
            case 2:
            {
                processClass = new Tvl1Method(samplePointSize,scanRadius);
                break;
            }
            #endif
            case 3:
            {
                if(useCuda){
                    #ifdef CUDA_SUPPORTED
                    processClass = new FastSpacedBMethod(samplePointSize,scanRadius,stepSize,cx,cy,fx,fy,k1,k2,k3,p1,p2);
                    #endif
                }else{
                    processClass = new BlockMethod(frameSize,samplePointSize,scanRadius,scanDiameter,scanCount,stepSize);
                }
                break;
            }
            case 4:
            {
                processClass = new FftMethod(frameSize,samplePointSize,max_px_speed_t);
                break;
            }
        case 5:
        {
            processClass = new FastSpacedBMMethod(samplePointSize,scanRadius,stepSize,cx,cy,fx,fy,k1,k2,k3,p1,p2);
            break;
        }

        }



        imPrev = cv::Mat(frameSize,frameSize,CV_8UC1);
        imPrev = cv::Scalar(0);
        processClass->setImPrev(imPrev);


        begin = ros::Time::now();



        //image_transport::ImageTransport iTran(node);

        VelocityPublisher = node.advertise<geometry_msgs::Twist>("/optFlow/velocity", 1);
        VelocityRawPublisher = node.advertise<geometry_msgs::Twist>("/optFlow/velocity_raw", 1);


        RangeSubscriber = node.subscribe(RangerPath,1,&OpticFlow::RangeCallback, this);

        if (useOdom){
            TiltSubscriber = node.subscribe("/uav5/mbzirc_odom/new_odom",1,&OpticFlow::CorrectTilt, this);
        }

        if (ImgCompressed){
            ImageSubscriber = node.subscribe(ImgPath, 1, &OpticFlow::ProcessCompressed, this);
        }else{
            ImageSubscriber = node.subscribe(ImgPath, 1, &OpticFlow::ProcessRaw, this);
        }

    }
    ~OpticFlow(){

    }

private:


    void RangeCallback(const sensor_msgs::Range& range_msg)
    {
        if (range_msg.range < 0.001) //zero
        {
            return;
        }

        currentRange = range_msg.range;
        if (!useOdom)
        {
            ros::Duration sinceLast = RangeRecTime -ros::Time::now();
            Zvelocity = (currentRange - prevRange)/sinceLast.toSec();
            trueRange = currentRange;
            RangeRecTime = ros::Time::now();
            prevRange = currentRange;
        }

    }

    void CorrectTilt(const nav_msgs::Odometry odom_msg)
    {
        tf::Quaternion bt;
        tf::quaternionMsgToTF(odom_msg.pose.pose.orientation,bt);
        tf::Matrix3x3(bt).getRPY(roll, pitch, yaw);
        Zvelocity = odom_msg.twist.twist.linear.z;

        angVel = cv::Point2d(odom_msg.twist.twist.angular.y,odom_msg.twist.twist.angular.x);


        if (currentRange > maxTerraRange)
        {
            trueRange = odom_msg.pose.pose.position.z;
        }
        else
        {
            trueRange = cos(pitch)*cos(roll)*currentRange;
        }
    }


    void ProcessCompressed(const sensor_msgs::CompressedImageConstPtr& image_msg)
    {
        cv_bridge::CvImagePtr image;
        image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
        Process(image);
    }

    void ProcessRaw(const sensor_msgs::ImageConstPtr& image_msg)
    {
        cv_bridge::CvImagePtr image;
        image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
        Process(image);
    }




    void Process(const cv_bridge::CvImagePtr image)
    {
        // First things first
        if (first)
        {
            if (ScaleFactor == 1)
            {
                int parameScale = image->image.cols/expectedWidth;
                fx = fx*parameScale;
                cx = cx*parameScale;
                fy = fy*parameScale;
                cy = cy*parameScale;
                k1 = k1*parameScale;
                k2 = k2*parameScale;
                k3 = k3*parameScale;
                p1 = p1*parameScale;
                p2 = p2*parameScale;

            }

            if(DEBUG){
                ROS_INFO("Source img: %dx%d", image->image.cols, image->image.rows);
            }

            first = false;
        }

        // Print out frequency
        ros::Duration dur = ros::Time::now()-begin;
        begin = ros::Time::now();
        if(DEBUG){
            ROS_INFO("freq = %fHz",1.0/dur.toSec());
        }


        // Scaling
        if (ScaleFactor != 1){
            cv::resize(image->image,imOrigScaled,cv::Size(image->image.size().width/ScaleFactor,image->image.size().height/ScaleFactor));
        }else{
            imOrigScaled = image->image.clone();
        }

        //ROS_INFO("Here 1");


        // Cropping
        if (!coordsAcquired)
        {
            imCenterX = imOrigScaled.size().width / 2;
            imCenterY = imOrigScaled.size().height / 2;
            xi = imCenterX - (frameSize/2);
            yi = imCenterY - (frameSize/2);
            frameRect = cv::Rect(xi,yi,frameSize,frameSize);
            midPoint = cv::Point2i((frameSize/2),(frameSize/2));
        }

        //ROS_INFO("Here 2");

        //  Converting color
        cv::cvtColor(imOrigScaled(frameRect),imCurr,CV_RGB2GRAY);

        // Call the method function
        std::vector<cv::Point2f> speeds = processClass->processImage(imCurr,gui,DEBUG,midPoint);


        // Check for wrong values
        speeds = removeNanPoints(speeds);

        if(speeds.size() <= 0){
            ROS_WARN("Processing function returned no valid points!");
            return;
        }

        // Print output
        if(DEBUG){
            for(uint i=0;i<speeds.size();i++){
                ROS_INFO("%d -> vxr = %f; vyr=%f",i,speeds[i].x,speeds[i].y);
            }
        }

        // Calculate real velocity
        multiplyAllPts(speeds,
                       -trueRange/(fx*dur.toSec()),
                       trueRange/(fy*dur.toSec())
                       );
        //vxm = -(out.x*(trueRange/fx))/dur.toSec();
        //vym = (out.y*(trueRange/fy))/dur.toSec();

        // Publish velocity without corrections, just an average...
        cv::Point2f rawAvg = pointMean(speeds);

        geometry_msgs::Twist velocity;
        velocity.linear.x = rawAvg.x;
        velocity.linear.y = rawAvg.y;
        velocity.linear.z = Zvelocity;
        velocity.angular.z = trueRange;
        VelocityRawPublisher.publish(velocity);

        // CORRECTIONS

        // camera rotation (within the construction) correction
        double phi = -(-gamma+1.570796326794897);

        //rotate2d(vxm,vym,phi);
        rotateAllPts(speeds,phi);

        // tilt correction
        //vxm = vxm + (tan(angVel.y*dur.toSec())*trueRange)/dur.toSec();
        //vym = vym + (tan(angVel.x*dur.toSec())*trueRange)/dur.toSec();
        addToAll(speeds, (tan(angVel.y*dur.toSec())*trueRange)/dur.toSec(),(tan(angVel.x*dur.toSec())*trueRange)/dur.toSec());

        // transform to global system
        phi = -(-yaw);

        //rotate2d(vxm,vym,phi);
        rotateAllPts(speeds,phi);

        // Bound speeds
        ros::Time timeNow = ros::Time::now();
        ros::Duration timeDif = timeNow - lastPublish;

        float max_sp_from_accel = lastVelocity + maxAccel*timeDif.toSec();

        speeds = getOnlyInAbsBound(speeds,0,maxSpeed); // bound according to max speed
        speeds = getOnlyInAbsBound(speeds,0,max_sp_from_accel); // bound according to max acceleration

        if(speeds.size() < 1){
            return;
        }

        // Do the Allsac/Ransac
        cv::Point2f out;
        if(Allsac){
            out = allsacMean(speeds,RansacThresholdRadSq);
        }else{
            out = ransacMean(speeds,RansacNumOfChosen,RansacThresholdRadSq,RansacNumOfIter);
        }

        vam = sqrt(getNormSq(out));
        vxm = out.x;
        vym = out.y;

        // Print out info
        if(DEBUG){
            ROS_INFO("vxm = %f; vym=%f; vzm=%f; vam=%f; range=%f; yaw=%f",vxm,vym,Zvelocity,vam,trueRange,yaw );
        }

        // Publish it
        velocity.linear.x = vxm;
        velocity.linear.y = vym;
        velocity.linear.z = Zvelocity;
        velocity.angular.z = trueRange;
        VelocityPublisher.publish(velocity);

        lastPublish = timeNow; // remember time from the point on some line higher because RANSAC and even ALLSAC can have very different processing time
        lastVelocity = vam;

        // Warn on wrong values
        if(abs(vxm) > 1000 || abs(vym) > 1000)
        {
            ROS_WARN("Suspiciously high velocity! vxm = %f; vym=%f; vzm=%f; vam=%f; range=%f\ntime=%f; rawRange=%f; yaw=%f pitch=%f; roll=%f\nangvelX=%f; angVelY=%f\nxp=%f; yp=%f",vxm,vym,Zvelocity,vam,trueRange,dur.toSec(),currentRange,yaw,pitch,roll,angVel.x,angVel.y,vxm,vym );
        }

    }

private:

    bool first;

    ros::Time RangeRecTime;

    ros::Subscriber ImageSubscriber;
    ros::Subscriber RangeSubscriber;
    ros::Publisher VelocityPublisher;    
    ros::Publisher VelocityRawPublisher;

    ros::Subscriber TiltSubscriber;


    cv::Mat imOrigScaled;
    cv::Mat imCurr;
    cv::Mat imPrev;

    double vxm, vym, vam;

    int imCenterX, imCenterY;    //center of original image
    int xi, xf, yi, yf; //frame corner coordinates
    cv::Point2i midPoint;
    bool coordsAcquired;
    cv::Rect frameRect;


    ros::Time begin;
    OpticFlowCalc *processClass;


    // Input arguments
    bool DEBUG;

    //std::vector<double> camRot;
    double gamma; // rotation of camera in the helicopter frame (positive)

    int expectedWidth;
    int ScaleFactor;

    int frameSize;
    int samplePointSize;

    int scanRadius;
    int scanDiameter;
    int scanCount;
    int stepSize;

    double cx,cy,fx,fy,s;
    double k1,k2,p1,p2,k3;

    bool gui, publish, useCuda, useOdom;
    int method;

    int numberOfBins;

    bool cameraRotated;

    int RansacNumOfChosen;
    int RansacNumOfIter;
    float RansacThresholdRadSq;
    bool Allsac;

    // Ranger & odom vars
    double currentRange;
    double trueRange;
    double prevRange;
    double Zvelocity;
    double roll, pitch, yaw;


    double max_px_speed_t;
    float maxSpeed;
    float maxAccel;
    bool checkAccel;
    ros::Time lastPublish;
    float lastVelocity;

    cv::Point2d angVel;



};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "optic_flow");
    ros::NodeHandle nodeA;

    OpticFlow of(nodeA);
    ros::spin();
    return 0;
}

