#define measureDistance 0.5

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Range.h>
#include <geometry_msgs/Pose2D.h>
using namespace std;
#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/gpuoptflow.hpp>
//#include <opencv2/gpulegacy.hpp>
//#include <opencv2/gpuimgproc.hpp>
//#include <time.h>

#include "optic_flow/FastSpacedBMOptFlow.h"


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

        ros::NodeHandle private_node_handle("~");
        private_node_handle.param("DEBUG", DEBUG, bool(false));

        private_node_handle.param("useCuda", useCuda, bool(false));
        private_node_handle.param("cudaMethod", cudaMethod, int(0));

        private_node_handle.param("FrameSize", frameSize, int(64));
        private_node_handle.param("SamplePointSize", samplePointSize, int(8));
        private_node_handle.param("ScanRadius", scanRadius, int(8));

        private_node_handle.param("gui", gui, bool(false));
        private_node_handle.param("publish", publish, bool(true));

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

        if ((frameSize % 2) == 1)
        {
            frameSize--;
        }
        scanDiameter = (2*scanRadius+1);
        scanCount = (scanDiameter*scanDiameter);
        xHist = new int[scanDiameter];
        yHist = new int[scanDiameter];

        imPrev = cv::Mat(frameSize,frameSize,CV_8UC1);
        imPrev = cv::Scalar(0);


        begin = ros::Time::now();



        //image_transport::ImageTransport iTran(node);

        VelocityPublisher = node.advertise<geometry_msgs::Pose2D>("/optFlow/velocity", 1);
        RangeSubscriber = node.subscribe(RangerPath,1,&OpticFlow::RangeCallback, this);
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

        cv::gpu::BroxOpticalFlow brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
        //lk =    cv::gpu::DensePyrLKOpticalFlow::create(cv::Size(7, 7));
        //cv::Ptr<cv::gpu::DensePyrLKOpticalFlow> lk;
        cv::gpu::FarnebackOpticalFlow farn;
        //cv::Ptr<cv::gpu::OpticalFlowDual_TVL1> tvl1;

        ros::Duration dur = ros::Time::now()-begin;
        ROS_INFO("freq = %fHz",1.0/dur.toSec());
        begin = ros::Time::now();


        ROS_INFO("res: %dX%d",image->image.size().width,image->image.size().height);

        if (ScaleFactor != 1)
            cv::resize(image->image,imOrigScaled,cv::Size(image->image.size().width/ScaleFactor,image->image.size().height/ScaleFactor));
        else
            imOrigScaled = image->image.clone();

        if (!coordsAcquired)
        {
            imCenterX = imOrigScaled.size().width / 2;
            imCenterY = imOrigScaled.size().height / 2;
            xi = imCenterX - (frameSize/2);
            yi = imCenterY - (frameSize/2);
            frameRect = cv::Rect(xi,yi,frameSize,frameSize);
            midPoint = cv::Point2i((frameSize/2),(frameSize/2));
        }

        cv::cvtColor(imOrigScaled(frameRect),imCurr,CV_RGB2GRAY);

        if (useCuda || (cudaMethod<4))
        {
            imPrev_g = cv::gpu::GpuMat(imPrev);
            imCurr_g = cv::gpu::GpuMat(imCurr);
            //flow_g = cv::gpu::GpuMat(imPrev.size(), CV_32FC2);
            onePixel_g = cv::gpu::GpuMat(1,1,CV_32SC1);

            ROS_INFO("method: %d",cudaMethod);
            if (cudaMethod == 0)
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
            else if (cudaMethod == 1)
                {
                    clock_t beginGPU = clock();
                    farn(imCurr_g, imPrev_g, flowX_g, flowY_g);
                    clock_t end = clock();
                      double elapsed_secs = double(end - beginGPU) / CLOCKS_PER_SEC;

                    cout << "Farn : " << elapsed_secs*1000 << " msec" << endl;

                    if (gui)
                        showFlow("Brox", flowX_g,flowY_g);
                }
            else if (cudaMethod == 2)
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
                ROS_INFO("vxr = %f; vyr=%f",outputX[0],outputY[0]);
                double vxm, vym, vam;
                vxm = outputX[0]*(currentRange/fx)/dur.toSec();
                vym = outputY[0]*(currentRange/fy)/dur.toSec();
                vam = sqrt(vxm*vxm+vym*vym);
                ROS_INFO("vxm = %f; vym=%f; vam=%f",vxm,vym,vam );

                geometry_msgs::Pose2D velocity;
                velocity.x = vxm;
                velocity.y = vym;
                VelocityPublisher.publish(velocity);VelocityPublisher;

            }
            else if (cudaMethod == 3)
            {
                //my method

                FastSpacedBMOptFlow(imCurr_g,imPrev_g, flowX_g,flowY_g,8,8,8);

            }

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
                        if (useCuda)
                        {

                           cv::gpu::absdiff(imCurr_g(cv::Rect(startPos,cv::Size(samplePointSize,samplePointSize))),
                                              imPrev_g(cv::Rect(startPos+cv::Point2i(j,i),
                                                       cv::Size(samplePointSize,samplePointSize))),
                                                                imDiff_g);
                           cv::Rect ROI(cv::Point2i(scanRadius,scanRadius)+cv::Point2i(i,j),cv::Size(1,1));
                           //ROS_INFO("ROI.x:%d, ROI.Y:%d, ROI.W:%d, ROI.H:%d",ROI.x, ROI.y,ROI.width, ROI.height );
                           //ROS_INFO("Mat.W:%d, Mat.H:%d",absDiffsMat_g.size().width, absDiffsMat_g.size().height );

                           onePixel_g = cv::gpu::GpuMat(absDiffsMat_g,ROI);
                           onePixel_g.setTo(cv::Scalar(cv::gpu::sum(imDiff_g)));
                           //onePixel_g = cv::gpu::GpuMat(absDiffsMat_g,cv::Rect(cv::Point2i(scanRadius,scanRadius)+cv::Point2i(j,i),cv::Size(1,1)));

                           //onePixel_g.setTo(cv::Scalar(cv::sum(cv::Mat(imDiff_g))));


                        }
                        else
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
                if (useCuda)
                {
                    cv::gpu::minMaxLoc(absDiffsMat_g, &min, &max, &min_loc, &max_loc);
                }
                else
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
        double vxm, vym, vam;
        vxm = refined.x*(currentRange/fx)/dur.toSec();
        vym = refined.y*(currentRange/fy)/dur.toSec();
        vam = sqrt(vxm*vxm+vym*vym);
        ROS_INFO("vxm = %f; vym=%f; vam=%f",vxm,vym,vam );

        geometry_msgs::Pose2D velocity;
        velocity.x = vxm;
        velocity.y = vym;
        VelocityPublisher.publish(velocity);VelocityPublisher;


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

    void RangeCallback(const sensor_msgs::Range& range_msg)
    {
                //ROS_INFO("here");
        currentRange = range_msg.range;
    }

    void drawOpticalFlow(const cv::Mat_<float>& flowx, const cv::Mat_<float>& flowy, cv::Mat& dst, float maxmotion = -1)
    {
        imView = imCurr.clone();

        for (int y = 0; y < flowx.rows; y+=samplePointSize)
        {
            for (int x = 0; x < flowx.cols; x+=samplePointSize)
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

    void showFlow(const char* name, const cv::gpu::GpuMat& d_flow_x, const cv::gpu::GpuMat& d_flow_y)
    {
        cv::Mat flowx(d_flow_x);
        cv::Mat flowy(d_flow_y);

        cv::Mat out;
        drawOpticalFlow(flowx, flowy, out, 10);

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


    /*int getHistMax(cv::Mat input)
    {
        cv::gpu::cal

        for (int i = 0;i<scanDiameter;i++)
        {
            xHist[i]=0;
            yHist[i]=0;
        }


    }*/

    bool DEBUG;

    ros::Subscriber ImageSubscriber;
    ros::Subscriber RangeSubscriber;
    ros::Publisher VelocityPublisher;


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

    int ScaleFactor;

    int frameSize;
    int samplePointSize;

    int scanRadius;
    int scanDiameter;
    int scanCount;

    double cx,cy,fx,fy,s;

    double currentRange;

    int imCenterX, imCenterY;    //center of original image
    int xi, xf, yi, yf; //frame corner coordinates
    cv::Point2i midPoint;
    bool coordsAcquired;
    cv::Rect frameRect;

    int *xHist;
    int *yHist;

    ros::Time begin;

    bool gui, publish, useCuda;
    int cudaMethod;

};
/*
class Viewer
{
public:
    Viewer(ros::NodeHandle& node)
    {
        ImageSubscriber = node.subscribe("/optFlow/diffImage", 1, &Viewer::Process, this);
    }

private:
    void Process(const sensor_msgs::ImageConstPtr& image_msg)
    {
        cv_bridge::CvImagePtr image;


        image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
        cv::imshow("main",image->image);

    }


    ros::Subscriber ImageSubscriber;
};
*/

int main(int argc, char** argv)
{
    ros::init(argc, argv, "optic_flow");
    ros::NodeHandle nodeA;

    OpticFlow of(nodeA);
    ros::spin();
    return 0;
}

