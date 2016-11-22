#include "../include/optic_flow/FastSpacedBMMethod_OCL.h"
#include <ostream>
#include <dirent.h>
#include "ros/package.h"

FastSpacedBMMethod::FastSpacedBMMethod(int i_samplePointSize,
                                     int i_scanRadius,
                                     int i_stepSize,
                                     int i_cx,int i_cy,int i_fx,int i_fy,
                                     int i_k1,int i_k2,int i_k3,int i_p1,int i_p2)
    {
    initialized = false;

    samplePointSize = i_samplePointSize;
    scanRadius = i_scanRadius;
    stepSize = i_stepSize;
    cx = i_cx;
    cy = i_cy;
    fx = i_fx;
    fy = i_fy;
    k1 = i_k1;
    k2 = i_k2;
    k3 = i_k3;
    p1 = i_p1;
    p2 = i_p2;

    cv::ocl::DevicesInfo devsInfo;
    if (cv::ocl::getOpenCLDevices(devsInfo))
    {
        for (int i=0; i<devsInfo.size(); i++)
        {
            std::cout << "Device " << i+1 << ": " << devsInfo[i]->deviceName << std::endl;
        }
    }
    else
    {
        std::cout << "No devices found." << std::endl;
        return;
    }

    FILE *program_handle;
    size_t program_size;
    //ROS_INFO((ros::package::getPath("optic_flow")+"/src/FastSpacedBMMethod.cl").c_str());
    program_handle = fopen((ros::package::getPath("optic_flow")+"/src/FastSpacedBMMethod.cl").c_str(),"r");
    if(program_handle == NULL)
    {
        std::cout << "Couldn't find the program file" << std::endl;
        return;
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    kernelSource = (char*)malloc(program_size + 1);
    kernelSource[program_size] = '\0';
    fread(kernelSource, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = new cv::ocl::ProgramSource("OptFlow",kernelSource);
    initialized = true;
    return ;

}

std::vector<cv::Point2f> FastSpacedBMMethod::processImage(cv::Mat imCurr_t,
                                              bool gui,
                                              bool debug,
                                             cv::Point midPoint_t)
{
    std::vector<cv::Point2f> outvec;

    if (!initialized)
    {
        std::cout << "Structure was not initialized; Returning.";
        return outvec;
    }
    midPoint = midPoint_t;

    int scanDiameter = (2*scanRadius)+1;
    int blockszX = samplePointSize+stepSize;
    int blockszY = samplePointSize+stepSize;

    imCurr = imCurr_t;

    imPrev_g.upload(imPrev);
    imCurr_g.upload(imCurr);

    std::size_t grid[3] = {(imPrev.cols-scanRadius*2)/blockszX,
                           (imPrev.rows-scanRadius*2)/blockszY,
                           1};
    std::size_t block[3] = {scanDiameter,scanDiameter,1};
    //std::size_t global[3] = {block[0]*8,block[1]*8,1};
    std::size_t global[3] = {grid[0]*block[0],grid[1]*block[1],1};
    std::size_t one[3] = {1,1,1};


    imflowX_g = cv::ocl::oclMat(cv::Size(grid[0],grid[1]),CV_8SC1);
    imflowX_g = cv::Scalar(-128);
    imflowY_g = cv::ocl::oclMat(cv::Size(grid[0],grid[1]),CV_8SC1);
    imflowY_g = cv::Scalar(-128);
    cl_int imSrcWidth_g = imCurr_g.step / imCurr_g.elemSize();
    cl_int imSrcOffset_g = imCurr_g.offset / imCurr_g.elemSize();
    cl_int imDstWidth_g = imflowX_g.step / imflowX_g.elemSize();
    cl_int imDstOffset_g = imflowX_g.offset/ imflowX_g.elemSize();
    cl_int samplePointSize_g = samplePointSize;
    cl_int stepSize_g = stepSize;
    cl_int scanRadius_g = scanRadius;
    cl_int scanDiameter_g = scanDiameter;

    int testval = 0;

    std::vector<std::pair<size_t , const void *> > args;
    args.clear();
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imPrev_g.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imCurr_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcOffset_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstOffset_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imflowX_g.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imflowY_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &samplePointSize_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &stepSize_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &scanRadius_g));

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
                                        *program,
                                        "OptFlow",
                                        global,
                                        block,
                                        args,
                                        1,
                                        0,
                                        NULL);
    signed char *outX_l = new signed char;
    signed char *outY_l = new signed char;
/*
    args.clear();
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imflowX_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstOffset_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &scanRadius_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &scanDiameter_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &outX_l ));

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
                                        *program,
                                        "Histogram",
                                        one,
                                        block,
                                        args,
                                        1,
                                        0,
                                        NULL);

    args.clear();
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imflowY_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstOffset_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &scanRadius_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &scanDiameter_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &outY_l ));

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
                                        *program,
                                        "Histogram",
                                        one,
                                        block,
                                        args,
                                        1,
                                        0,
                                        NULL);
*/
    cv::Mat flowx(cv::Size(grid[0],grid[1]),CV_8SC1);
    cv::Mat flowy(cv::Size(grid[0],grid[1]),CV_8SC1);
    imflowX_g.download(flowx);
    imflowY_g.download(flowy);
    showFlow(flowx,flowy,*outX_l, *outY_l);
    if (debug)
    {
       // ROS_INFO("out: %dx%d",outX_l.cols,outX_l.rows);
    }
    if (gui)
    {
       // showFlow(flowX_l, flowY_l;, outputX, outputY);
    }

    imPrev = imCurr.clone();

    outvec.push_back(cv::Point2f((float)*outX_l,(float)*outY_l));
    return outvec;

}

void FastSpacedBMMethod::showFlow(const cv::Mat flowx, const cv::Mat flowy, cl_char vXin, cl_char vYin)
{
    cv::Mat out;
    drawOpticalFlow(flowx, flowy, out, 10, stepSize);

    cv::line(imView,
              midPoint,
              midPoint+cv::Point2i(((int)vXin*4),((int)vYin*4)),
              cv::Scalar(255),2);

    cv::imshow("Main", imView);
    cv::waitKey(10);
}

void FastSpacedBMMethod::drawOpticalFlow(const cv::Mat_<signed char>& flowx, const cv::Mat_<signed char>& flowy, cv::Mat& dst, float maxmotion,
                     int step)
{
    imView = imCurr.clone();

    for (int y = 0; y < flowx.rows; y++)
    {
        for (int x = 0; x < flowx.cols; x++)
        {
            if ((abs(flowx(y, x)) > scanRadius) || (abs(flowy(y, x))> scanRadius))
            {
                //ROS_WARN("Flow out of bounds: X:%d, Y:%d",flowx(y, x),flowy(y, x));
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

