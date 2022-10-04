#ifndef __YOLO7_HPP__
#define __YOLO7_HPP__ 
#include "utils.hpp"

class Yolo7
{
private:
    Logger gLogger;
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context;
    void* buffers[4];
    const int inputIndex = 0;
    const int outputIndex1 = 1;
    const int outputIndex2 = 2;
    const int outputIndex3 = 3;
    float *blob;
    float *prob1,*prob2,*prob3;
    cudaStream_t stream;
    int img_w;
    int img_h;
    
public:
    Yolo7(std::string engine_file_path);
    ~Yolo7();
    void detect(const cv::Mat &img,std::vector<Object> &objects);

};


#endif
