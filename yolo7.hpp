#ifndef __YOLO7_HPP__
#define __YOLO7_HPP__ 
#include "utils.hpp"


#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.5

class Yolo7
{
private:
    Logger gLogger;
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context;
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine;
    std::vector<void*> buffers;
    const int inputIndex = 0;
    const int outputIndex1 = 1;
    const int outputIndex2 = 2;
    const int outputIndex3 = 3;
    float *blob;
    float *prob1,*prob2,*prob3;
    cudaStream_t stream;
    int img_w;
    int img_h;
    cv::Mat re;
    float scale;
    size_t output_size1,output_size2,output_size3;

public:
    Yolo7(std::string engine_file_path, int img_w, int img_h);
    ~Yolo7();
    void detect(const cv::Mat &img,std::vector<Object> &objects);

};


#endif
