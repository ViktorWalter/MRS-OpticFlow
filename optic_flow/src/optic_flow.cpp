#define maxTerraRange 8.0

#include <ros/ros.h>
#include <tf/tf.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
using namespace std;
#include <opencv2/gpu/gpu.hpp>

//#include <opencv2/gpuoptflow.hpp>
//#include <opencv2/gpulegacy.hpp>
//#include <opencv2/gpuimgproc.hpp>
//#include <time.h>

#include "optic_flow/FastSpacedBMOptFlow.h"

#include "optic_flow/BlockMethod.h"
#include "optic_flow/FftMethod.h"
#include "optic_flow/BroxMethod.h"
#include "optic_flow/FarnMethod.h"
#include "optic_flow/FastSpacedBMethod.h"
#include "optic_flow/Tvl1Method.h"
#include "optic_flow/OpticFlowCalc.h"

//#define CUDA_SUPPORTED

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
        private_node_handle.param("CameraImageCompressed", ImgCompressed, bool(false));

        private_node_handle.param("RangerDataPath", RangerPath, std::string("/uav/terarangerone"));

        private_node_handle.param("ScaleFactor", ScaleFactor, int(1));

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
        xHist = new int[scanDiameter];
        yHist = new int[scanDiameter];


        private_node_handle.param("cameraRotated", cameraRotated, bool(true));
        private_node_handle.getParam("camera_rotation_matrix/data", camRot);

        if (useCuda)
        {
            ResetCudaDevice();
            imPrev_g.create(imPrev.size(),imPrev.type());
            imCurr_g.create(imPrev.size(),imPrev.type());
        }


        if(useCuda && method != 3){
            ROS_WARN("Method does not have cuda/noncuda version.");
        }

        useProcessClass = false;
        if(1){
            switch(method){
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
                case 3:
                {
                    if(useCuda){
                        processClass = new FastSpacedBMethod(samplePointSize,scanRadius,stepSize,cx,cy,fx,fy,k1,k2,k3,p1,p2);
                    }else{
                        processClass = new BlockMethod(frameSize,samplePointSize,scanRadius,scanDiameter,scanCount,stepSize);
                    }
                    break;
                }
                case 4:
                {
                    processClass = new  FftMethod(frameSize,samplePointSize,numberOfBins);
                    break;
                }

            }
            useProcessClass = true;

        }


        imPrev = cv::Mat(frameSize,frameSize,CV_8UC1);
        imPrev = cv::Scalar(0);
        processClass->setImPrev(imPrev);


        begin = ros::Time::now();



        //image_transport::ImageTransport iTran(node);

        VelocityPublisher = node.advertise<geometry_msgs::Twist>("/optFlow/velocity", 1);
        VelocityRawPublisher = node.advertise<geometry_msgs::Twist>("/optFlow/velocity_raw", 1);


        RangeSubscriber = node.subscribe(RangerPath,1,&OpticFlow::RangeCallback, this);

        if (useOdom)
            TiltSubscriber = node.subscribe("/uav5/mbzirc_odom/new_odom",1,&OpticFlow::CorrectTilt, this);

        if (ImgCompressed)
            ImageSubscriber = node.subscribe(ImgPath, 1, &OpticFlow::ProcessCompressed, this);
        else
            ImageSubscriber = node.subscribe(ImgPath, 1, &OpticFlow::ProcessRaw, this);


    }
    ~OpticFlow()
    {
        delete[] xHist;
        delete[] yHist;
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

            if(DEBUG)
                ROS_INFO("Source img: %dx%d", image->image.cols, image->image.rows);
            first = false;
        }

#ifdef CUDA_SUPPORTED
        cv::gpu::BroxOpticalFlow brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
        cv::gpu::FarnebackOpticalFlow farn;
#endif

        ros::Duration dur = ros::Time::now()-begin;
        if(DEBUG)
            ROS_INFO("freq = %fHz",1.0/dur.toSec());
        begin = ros::Time::now();



        if (ScaleFactor != 1)
            cv::resize(image->image,imOrigScaled,cv::Size(image->image.size().width/ScaleFactor,image->image.size().height/ScaleFactor));
        else
            imOrigScaled = image->image.clone();

        ROS_INFO("Here 1");

        if (!coordsAcquired)
        {
            imCenterX = imOrigScaled.size().width / 2;
            imCenterY = imOrigScaled.size().height / 2;
            xi = imCenterX - (frameSize/2);
            yi = imCenterY - (frameSize/2);
            frameRect = cv::Rect(xi,yi,frameSize,frameSize);
            midPoint = cv::Point2i((frameSize/2),(frameSize/2));
        }

        ROS_INFO("Here 2");

        cv::cvtColor(imOrigScaled(frameRect),imCurr,CV_RGB2GRAY);
        ROS_INFO("Here 3");

        if(useProcessClass){
            cv::Point2f out = processClass->processImage(imCurr,gui,DEBUG,midPoint);

            if(DEBUG)
                ROS_INFO("vxr = %f; vyr=%f",out.x,out.y);
            double vxm, vym, vam;

            vxm = -(out.x*(trueRange/fx))/dur.toSec();
            vym = (out.y*(trueRange/fy))/dur.toSec();

            geometry_msgs::Twist velocity;
            velocity.linear.x = vxm;
            velocity.linear.y = vym;
            velocity.linear.z = Zvelocity;
            velocity.angular.z = trueRange;
            VelocityRawPublisher.publish(velocity);

            if (cameraRotated)
            {
                double vxm_n = camRot[0]*vxm + camRot[1]*vym;
                double vym_n = camRot[2]*vxm + camRot[3]*vym;

                vxm = vxm_n;
                vym = vym_n;
            }

            vxm = vxm + (tan(angVel.y*dur.toSec())*trueRange)/dur.toSec();
            vym = vym + (tan(angVel.x*dur.toSec())*trueRange)/dur.toSec();

            //angular vel. corr (not with Z ax.)

            vam = sqrt(vxm*vxm+vym*vym);
            if(DEBUG)
                ROS_INFO("vxm = %f; vym=%f; vzm=%f; vam=%f; range=%f",vxm,vym,Zvelocity,vam,trueRange );

            velocity.linear.x = vxm;
            velocity.linear.y = vym;
            velocity.linear.z = Zvelocity;
            velocity.angular.z = trueRange;
            VelocityPublisher.publish(velocity);


        }else{

        if (useCuda && (method<4))
        {
#ifdef CUDA_SUPPORTED
            imPrev_g.upload(imPrev);
            imCurr_g.upload(imCurr);


            //ROS_INFO("Here");
            //ROS_INFO("method: %d",method);
            if (method == 0)
            {
                cv::gpu::GpuMat imPrev_gf;
                cv::gpu::GpuMat imCurr_gf;

                imPrev_g.convertTo(imPrev_gf, CV_32F, 1.0 / 255.0);
                imCurr_g.convertTo(imCurr_gf, CV_32F, 1.0 / 255.0);



                clock_t beginGPU = clock();
                brox(imCurr_gf, imPrev_gf, flowX_g,flowY_g);
                clock_t end = clock();
                double elapsed_secs = double(end - beginGPU) / CLOCKS_PER_SEC;

                cout << "Brox : " << elapsed_secs*1000 << " msec" << endl;

                if (gui)
                    showFlow("Brox", flowX_g,flowY_g);
            }
            else if (method == 1)
                {
                    clock_t beginGPU = clock();
                    farn(imCurr_g, imPrev_g, flowX_g, flowY_g);
                    clock_t end = clock();
                      double elapsed_secs = double(end - beginGPU) / CLOCKS_PER_SEC;

                    cout << "Farn : " << elapsed_secs*1000 << " msec" << endl;

                    if (gui)
                        showFlow("Brox", flowX_g,flowY_g);
                }
            else if (method == 2)
            {
                //cv::gpu::GpuMat buf_g;

                clock_t beginGPU = clock();
                fastBM( imCurr_g,imPrev_g, flowX_g,flowY_g,8);
                clock_t end = clock();
                  double elapsed_secs = double(end - beginGPU) / CLOCKS_PER_SEC;

                cout << "BM : " << elapsed_secs*1000 << " msec" << endl;

                if (gui)
                    showFlow("TVL1", flowX_g, flowY_g);


                cv::Scalar outputX;
                cv::Scalar outputY;
                outputX = cv::gpu::sum(flowX_g,buffer_g)/(double)flowX_g.size().area();
                outputY = cv::gpu::sum(flowY_g,buffer_g)/(double)flowY_g.size().area();

                //cv::Point2f refined = Refine(imCurr,imPrev,cv::Point2i(outputX,outputY),2);

                //ROS_INFO("vxu = %d; vyu=%d",outputX,outputY);
                //ROS_INFO("vxr = %f; vyr=%f",outputX[0],outputY[0]);
                double vxm, vym, vam;
                vxm = outputX[0]*(trueRange/fx)/dur.toSec();
                vym = outputY[0]*(trueRange/fy)/dur.toSec();
                vam = sqrt(vxm*vxm+vym*vym);
                ROS_INFO("vxm = %f; vym=%f; vam=%f",vxm,vym,vam );

                geometry_msgs::Twist velocity;
                velocity.linear.x = vxm;
                velocity.linear.y = vym;
                velocity.linear.z = Zvelocity;
                VelocityPublisher.publish(velocity);

            }
            else if (method == 3) {
                //my method
                signed char outputX;
                signed char outputY;

                cv::Mat outXconv;
                cv::Mat outYconv;

                FastSpacedBMOptFlow(imCurr_g,imPrev_g, flowX_g,flowY_g,samplePointSize,stepSize,scanRadius,
                                    cx, cy, fx,fy, k1, k2, k3, p1, p2,
                                    outputX,
                                    outputY
                                    );

                if (DEBUG)
                {
                    ROS_INFO("out: %dx%d",flowX_g.cols,flowX_g.rows);
                    //imView = cv::Mat(imCurr_g);


                }
                if (gui)
                {
                    showFlow("TVL1", flowX_g, flowY_g, outputX, outputY);
                }



                //cv::Point2f refined = Refine(imCurr,imPrev,cv::Point2i(outputX,outputY),2);

                ROS_INFO("vxr = %d; vyr=%d",outputX,outputY);
                double vxm, vym, vzm, vam;
                vxm = outputX*(trueRange/fx)/dur.toSec();
                vym = outputY*(trueRange/fy)/dur.toSec();
                vam = sqrt(vxm*vxm+vym*vym);

                ROS_INFO("vxm = %f; vym=%f; vzm=%f; vam=%f",vxm,vym,Zvelocity,vam );

                geometry_msgs::Twist velocity;
                velocity.linear.x = vxm;
                velocity.linear.y = vym;
                velocity.linear.z = Zvelocity;
                velocity.angular.z = trueRange;
                VelocityPublisher.publish(velocity);



            }
#endif
        }
        else
        {

         imView = imCurr.clone();

        for (int i = 0;i<scanDiameter;i++)
        {
            xHist[i]=0;
            yHist[i]=0;
        }


        int maxSamplesSide = (frameSize-scanRadius*2)/(samplePointSize);
        cv::Point2i startPos;
        if (useCuda)
        {
            imPrev_g.upload(imPrev);
            imCurr_g.upload(imCurr);
        }
        for (int m = 0; m<maxSamplesSide; m++)
        {
            for (int n = 0; n<maxSamplesSide; n++)
                {
                startPos = cv::Point2i(n*(samplePointSize)+scanRadius,m*(samplePointSize)+scanRadius);
                if (useCuda)
                    absDiffsMat_g = cv::gpu::GpuMat(scanDiameter,scanDiameter,CV_32S);
                else
                    absDiffsMat = cv::Mat(scanDiameter,scanDiameter,CV_32S);

                int index = 0;

                for (int i = -scanRadius;i<=scanRadius;i++)
                {

                    for (int j = -scanRadius;j<=scanRadius;j++)
                    {                        
                        clock_t beginGPU = clock();
                        {    cv::absdiff(
                                    imCurr(cv::Rect(startPos,cv::Size(samplePointSize,samplePointSize))),
                                    imPrev(cv::Rect(startPos+cv::Point2i(j,i),
                                                    cv::Size(samplePointSize,samplePointSize))),
                                    imDiff
                                    );
                            absDiffsMat.at<int32_t>(cv::Point2i(scanRadius,scanRadius)+cv::Point2i(j,i)) = cv::sum(imDiff)[0];
                        }

                        clock_t endGPU = clock();
                        double elapsed_secs = double(endGPU - beginGPU) / CLOCKS_PER_SEC;
                        if (DEBUG)
                             ROS_INFO("calc time : %f msec",elapsed_secs*1000);
                        index++;
                    }
                }
                clock_t beginGPU = clock();
                double min, max;
                cv::Point min_loc, max_loc;
                {
                    cv::minMaxLoc(absDiffsMat, &min, &max, &min_loc, &max_loc);
                }
                xHist[min_loc.x]++;
                yHist[min_loc.y]++;
                clock_t endGPU = clock();
                double elapsed_secs = double(beginGPU-endGPU) / CLOCKS_PER_SEC;
                if (DEBUG)
                ROS_INFO("Hist : %f msec", elapsed_secs*1000);

                if (gui)
                    cv::line(imView,
                         startPos+cv::Point2i(samplePointSize/2,samplePointSize/2),
                         startPos+cv::Point2i(samplePointSize/2,samplePointSize/2)+min_loc-cv::Point2i(scanRadius,scanRadius),
                         cv::Scalar(255));

                }
        }

        int outputX = std::distance(xHist, std::max_element(xHist, xHist + scanDiameter))-scanRadius;
        int outputY = std::distance(yHist, std::max_element(yHist, yHist + scanDiameter))-scanRadius;        
        //ROS_INFO("x = %d; y = %d\n",outputX,outputY);

        cv::Point2f refined = Refine(imCurr,imPrev,cv::Point2i(outputX,outputY),2);

        //ROS_INFO("vxu = %d; vyu=%d",outputX,outputY);
        ROS_INFO("vxr = %f; vyr=%f",refined.x,refined.y);
        double vxm, vym, vzm, vam;
        vxm = refined.x*(trueRange/fx)/dur.toSec();
        vym = refined.y*(trueRange/fy)/dur.toSec();
        vam = sqrt(vxm*vxm+vym*vym);
        ROS_INFO("vxm = %f; vym=%f; vzm=%f; vam=%f",vxm,vym,Zvelocity,vam );

        geometry_msgs::Twist velocity;
        velocity.linear.x = vxm;
        velocity.linear.y = vym;
        velocity.linear.z = Zvelocity;
        VelocityPublisher.publish(velocity);


        if (gui)
        {
            cv::line(imView,
                      midPoint,
                      midPoint+cv::Point2i((int)(refined.x*4),(int)(refined.y*4)),
                      cv::Scalar(255),2);
            cv::imshow("main",imView);
            cv::waitKey(10);
        }

        //ImagePublisher.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", imDiff).toImageMsg());


        //imCurr.at<uchar>((frameSize/2)+locations[bestIndex].location.x,(frameSize/2)+locations[bestIndex].location.y) = 255;
        //cv::line(imCurr);
        }

        imPrev = imCurr.clone();

        }




    }

    cv::Point2f Refine(cv::Mat imCurr, cv::Mat imPrev, cv::Point2i fullpixFlow,int passes)
    {

        cv::Mat imCurr2x = imCurr.clone();;
        cv::Mat imPrev2x = imPrev.clone();
        cv::Mat imDiff2x;

        cv::Point2i totalOffset = fullpixFlow;

        int pixScale = 1;

        for (int i=1; i<=passes;i++)
        {
            pixScale = pixScale*2;
            totalOffset = totalOffset*2;

            cv::resize(imCurr2x,imPrev2x,cv::Size(imPrev.size().width*2,imPrev.size().height*2)); //optimalizuj -uloz aj neskreslene
            cv::resize(imCurr2x,imCurr2x,cv::Size(imCurr.size().width*2,imCurr.size().height*2));

            cv::Point2i startpoint;
            if ((totalOffset.x<0) && (totalOffset.y<0))
                startpoint = cv::Point2i(-totalOffset.x+1,-totalOffset.y+1);
            else if  ((totalOffset.x<0) && (totalOffset.y>=0))
                startpoint = cv::Point2i(-totalOffset.x+1,1);
            else if ((totalOffset.x>=0) && (totalOffset.y<0))
                startpoint = cv::Point2i(1,-totalOffset.y+1);
            else
                startpoint = cv::Point2i(1,1);

            cv::Size2i cutoutSize = cv::Size2i(imCurr2x.size().width-(abs(totalOffset.x)+2),
                                               imCurr2x.size().height-(abs(totalOffset.y)+2));




            absDiffsMatSubpix = cv::Mat(3,3,CV_32S);
            for (int m = -1; m<=1; m++)
            {
                for (int n = -1; n<=1; n++)
                {
                    //ROS_INFO("m=%d, n=%d, scale=%d, tx=%d, ty=%d",m,n,pixScale,totalOffset.x,totalOffset.y);
                    //ROS_INFO("spx=%d, spy=%d, szx=%d, szy=%d",startpoint.x,startpoint.y,cutoutSize.width,cutoutSize.height);

                    cv::absdiff(
                                imCurr2x(cv::Rect(cv::Point2i(1,1), cutoutSize)),
                                imPrev2x(cv::Rect(startpoint+cv::Point2i(n,m), cutoutSize)),
                                imDiff2x
                                );

                    absDiffsMatSubpix.at<int32_t>(cv::Point2i(1,1)+cv::Point2i(n,m)) = cv::sum(imDiff2x)[0];
                }
            }


            double min, max;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(absDiffsMatSubpix, &min, &max, &min_loc, &max_loc);

            totalOffset = totalOffset + min_loc - cv::Point2i(1,1);
        }
        cv::Point2f output = cv::Point2f(totalOffset.x/(float)pixScale,totalOffset.y/(float)pixScale);

        return output;
    }



    void drawOpticalFlow(const cv::Mat_<float>& flowx, const cv::Mat_<float>& flowy, cv::Mat& dst, float maxmotion = -1)
    {
        imView = imCurr.clone();

        for (int y = 0; y < flowx.rows; y+=(samplePointSize))
        {
            for (int x = 0; x < flowx.cols; x+=(samplePointSize))
            {
                cv::Point2i startPos(x,y);
                cv::Point2f u(flowx(y, x), flowy(y, x));
                cv::line(imView,
                     startPos+cv::Point2i(samplePointSize/2,samplePointSize/2),
                     startPos+cv::Point2i(samplePointSize/2,samplePointSize/2)+cv::Point2i(u.x,u.y),
                     cv::Scalar(255));

            }
        }
        dst = imView;
    }

    void drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion,
                         int step)
    {
        imView = imCurr.clone();

        for (int y = 0; y < flowx.rows; y++)
        {
            for (int x = 0; x < flowx.cols; x++)
            {
                if ((abs(flowx(y, x)) > scanRadius) || (abs(flowy(y, x))> scanRadius))
                {
                    ROS_WARN("Flow out of bounds: X:%d, Y:%d",flowx(y, x),flowy(y, x));
                    //continue;
                }
                cv::Point2i startPos(x*(step+samplePointSize)+(samplePointSize/2+scanRadius),
                                     y*(step+samplePointSize)+(samplePointSize/2+scanRadius));

                cv::Point2i u(flowx(y, x), flowy(y, x));
                cv::line(imView,
                     startPos,
                     startPos+u,
                     cv::Scalar(255));

            }
        }
        dst = imView;
    }

    void showFlow(const char* name, const cv::gpu::GpuMat& d_flow)
    {
        cv::gpu::GpuMat planes[2];
        cv::gpu::split(d_flow, planes);

        cv::Mat flowx(planes[0]);
        cv::Mat flowy(planes[1]);

        cv::Mat out;
        drawOpticalFlow(flowx, flowy, out, 10);

        int vx = getHistMaxGPU(planes[0]);
        int vy = getHistMaxGPU(planes[1]);


        cv::line(imView,
                  midPoint,
                  midPoint+cv::Point2i((int)(vx*4),(int)(vy*4)),
                  cv::Scalar(255),2);

        cv::imshow("Main", imView);
        cv::waitKey(10);
    }

    void showFlow(const char* name, const cv::gpu::GpuMat flowx_g, const cv::gpu::GpuMat flowy_g, signed char vXin, signed char vYin)
    {
        cv::Mat flowx = cv::Mat(flowx_g);
        cv::Mat flowy = cv::Mat(flowy_g);

        cv::Mat out;
        drawOpticalFlow(flowx, flowy, out, 10, stepSize);

        cv::line(imView,
                  midPoint,
                  midPoint+cv::Point2i(((int)vXin*4),((int)vYin*4)),
                  cv::Scalar(255),2);

        cv::imshow("Main", imView);
        cv::waitKey(10);
    }

    int getHistMaxGPU(cv::gpu::GpuMat input)
    {
        cv::gpu::GpuMat inputConv;
         input.convertTo(inputConv,CV_8UC1,1,scanRadius);
         cv::gpu::GpuMat Hist;
         Hist.setTo(cv::Scalar(0));
         double min, max;
         cv::Point min_loc, max_loc;
         cv::gpu::calcHist(inputConv,Hist);
         cv::gpu::minMaxLoc(Hist, &min, &max, &min_loc, &max_loc);
         cv::imshow("hist",cv::Mat(Hist));
        return max_loc.x-scanRadius;

    }

    void showFlow(const char* name, const cv::gpu::GpuMat& d_flow_x, const cv::gpu::GpuMat& d_flow_y)
    {
        cv::Mat flowx(d_flow_x);
        cv::Mat flowy(d_flow_y);

        cv::Mat out;
        drawOpticalFlow(flowx, flowy, out, 10, 0);

        //int vx = getHistMaxGPU(d_flow_x);
        //int vy = getHistMaxGPU(d_flow_y);
        double vx, vy;

        vx = cv::mean(flowx)[0];
        vy = cv::mean(flowy)[0];
        ROS_INFO("vx = %f; vy=%f",vx,vy);

        cv::line(imView,
                  midPoint,
                  midPoint+cv::Point2i((int)(vx*4),(int)(vy*4)),
                  cv::Scalar(255),2);

        cv::imshow("Main", imView);
        cv::waitKey(10);
    }


    void GpuSum(const cv::gpu::GpuMat& input, cv::gpu::GpuMat& output)
    {
    output.setTo(cv::Scalar(0));
        for (int i = 0; i<input.rows;i++)
        {
            for (int j = 0; j< input.cols; j++)
            {
                cv::gpu::GpuMat onePixel_g_t(input,cv::Rect(j,i,1,1));
                cv::gpu::add(onePixel_g_t,output,output);
            }
        }
        return;
    }







    bool DEBUG;
    bool first;

    ros::Time RangeRecTime;

    ros::Subscriber ImageSubscriber;
    ros::Subscriber RangeSubscriber;
    ros::Publisher VelocityPublisher;    
    ros::Publisher VelocityRawPublisher;

    ros::Subscriber TiltSubscriber;


    cv::Mat imOrigScaled;
    cv::Mat imCurr;
    cv::Mat imDiff;
    cv::Mat imPrev;

    cv::gpu::GpuMat imCurr_g;
    cv::gpu::GpuMat imPrev_g;
    cv::gpu::GpuMat imDiff_g;
    cv::gpu::GpuMat flowX_g;
    cv::gpu::GpuMat flowY_g;
    cv::gpu::GpuMat absDiffsMat_g;
    cv::gpu::GpuMat onePixel_g;
    cv::gpu::GpuMat onePixel_g_t;
    cv::gpu::GpuMat buffer_g;


    cv::gpu::FastOpticalFlowBM fastBM;

    cv::Mat imView;
    cv::Mat absDiffsMat;
    cv::Mat absDiffsMatSubpix;

    std::vector<double> camRot;

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

    double currentRange;
    double trueRange;
    double prevRange;
    double Zvelocity;
    double roll, pitch, yaw;

    cv::Point2d angVel;



    int imCenterX, imCenterY;    //center of original image
    int xi, xf, yi, yf; //frame corner coordinates
    cv::Point2i midPoint;
    bool coordsAcquired;
    cv::Rect frameRect;

    int *xHist;
    int *yHist;

    ros::Time begin;

    bool gui, publish, useCuda, useOdom;
    int method;

    int numberOfBins;

    OpticFlowCalc *processClass;

    bool useProcessClass;

    bool cameraRotated;


};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "optic_flow");
    ros::NodeHandle nodeA;

    OpticFlow of(nodeA);
    ros::spin();
    return 0;
}

