#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <string>

#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#include <fstream>
#include <string>




struct  TRTDestroy
{
    template<class T>
    void operator()(T* obj) const 
    {
        if (obj)
            // obj->destroy();
            delete obj;
    }
};



#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.5
#define BBOX_CONF_THRESH 0.5

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_W = 640;
static const int INPUT_H = 384;
static const int NUM_CLASSES = 80;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "o0";
static Logger gLogger;

cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    
    return out;
}

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float IoU = inter_area / union_area;
            if(IoU > nms_threshold)
            {
                keep = 0;

            }
            // if (inter_area / union_area > nms_threshold)
            //     keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
{

    const int num_anchors = grid_strides.size();
    // cout<<"num_anchors "<<num_anchors<<endl;
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}
/*
static void generate_yolo7_proposals(float* feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    for(int i=0; i < 11520;i++)
    {
        float x = feat_blob[i*85+0];
        float y = feat_blob[i*85+1];
        float w = feat_blob[i*85+2];
        float h = feat_blob[i*85+3];
        float obj_conf = feat_blob[i*85+4];
        std::vector<float> classes_scores;
        for(int j = 0;j <80; j++)
            classes_scores.push_back(feat_blob[i*85+5+j]);

        auto max_score = std::max_element(classes_scores.begin() , classes_scores.end()); 
        int argmaxVal = distance(classes_scores.begin(), max_score);


        float prob = classes_scores[argmaxVal]*obj_conf;
        if (prob > prob_threshold)
        {
            Object obj;
            obj.prob = prob;
            obj.label = argmaxVal;
            obj.rect = cv::Rect_<float>(x,y,w,h);
            objects.push_back(obj);
        }

    }
}*/
void generete_proposal_scale(float* feat_blob, float prob_threshold, std::vector<Object>& objects,int nx, int ny,float stride,float * anchor_grid_w,float * anchor_grid_h)
{
    for(int i1=0;i1<3;i1++)
    for(int i2=0;i2<ny;i2++) // nx = 80 stride = 8 
    for(int i3=0;i3<nx;i3++) // ny = 48
    // for(int i4=0;i4<85;i4++)
    {
        int i4 = 0;
        int this_index = i3*85 + i2*85*nx + i1*85*nx*ny;
        float x = feat_blob[i4+this_index];
        x = (x*2-0.5+i3)*stride ; 
        i4++;
        float y = feat_blob[i4+this_index];
        y = (y*2-0.5+i2)*stride; 
        i4++;
        float w = feat_blob[i4+this_index];
        w = w*2*w*2*anchor_grid_w[i1];
        i4++;
        float h = feat_blob[i4+this_index];
        h = h*2*h*2*anchor_grid_h[i1];
        i4++;
        float obj_conf = feat_blob[i4+this_index];
        i4++;
        
        // cout<<"prob: "<<prob<<endl   ;
        if (obj_conf > prob_threshold)
        {
            std::vector<float> classes_scores;
            for(;i4 <85; i4++)
                classes_scores.push_back(feat_blob[i4+this_index]);

            // cout<<"classes_scores: "s<<classes_scores.size()<<endl   ;

            auto max_score = std::max_element(classes_scores.begin() , classes_scores.end()); 
            int argmaxVal = distance(classes_scores.begin(), max_score);


            float prob = classes_scores[argmaxVal]*obj_conf;

            Object obj;
            obj.prob = prob;
            obj.label = argmaxVal;
            float x0 = x - w * 0.5f;
            float y0 = y - h * 0.5f;
            obj.rect = cv::Rect_<float>(x0,y0,w,h);
            objects.push_back(obj);
        }
    }
}
static void generate_yolo7_proposals(float* feat_blob1,float* feat_blob2,float* feat_blob3, float prob_threshold, std::vector<Object>& objects)
{
    float anchor_grid_w_1[] = {12.0,19.0,40.0};
    float anchor_grid_h_1[] = {16.0,36.0,28.0};

    float anchor_grid_w_2[] = {36,76,72};
    float anchor_grid_h_2[] = {75,55,146};

    float anchor_grid_w_3[] = {142,192,459};
    float anchor_grid_h_3[] = {110,243,401};

    generete_proposal_scale(feat_blob1,prob_threshold,objects,80,48,8,anchor_grid_w_1,anchor_grid_h_1);
    generete_proposal_scale(feat_blob2,prob_threshold,objects,40,24,16,anchor_grid_w_2,anchor_grid_h_2);
    generete_proposal_scale(feat_blob3,prob_threshold,objects,20,12,32,anchor_grid_w_3,anchor_grid_h_3);

    // cout<<"len objects "<<objects.size()<<endl;
    // std::ofstream out_file{"../objects.txt"};

    // for (const auto & o:objects)
    // {
    //     out_file<<o.rect.x<<","<<o.rect.y<<","<<o.rect.width<<","<<o.rect.height<<","<<o.label<<","<<o.prob<<endl;
    // }


    // out_file.close();

}

float* blobFromImage(cv::Mat& img){
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c]/255.0f;
            }
        }
    }
    return blob;
}


static void decode_outputs(float* prob1,float* prob2,float* prob3, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        // std::vector<int> strides = {8, 16, 32};
        // std::vector<GridAndStride> grid_strides;
        // generate_grids_and_stride(strides, grid_strides);


        // generate_yolox_proposals(grid_strides, prob1,  BBOX_CONF_THRESH, proposals);
        generate_yolo7_proposals(prob1, prob2, prob3,  BBOX_CONF_THRESH, proposals);

        // qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();


        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    // cv::imwrite("det_res.jpg", image);
    // fprintf(stderr, "save vis file\n");
    // cv::resize(image,image,cv::Size(2560,1440));
    cv::imshow("image", image); 
    
    cv::waitKey(1); 
}


void doInference(IExecutionContext& context, float* input, float* output1, const int output_size1, float* output2, const int output_size2, float* output3, const int output_size3, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // cout<<"engine.getNbBindings() "<<engine.getNbBindings()<<endl;
    assert(engine.getNbBindings() == 4); // it must be 4
    void* buffers[4];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex1 = 1 ;// engine.getBindingIndex(OUTPUT_BLOB_NAME);
    const int outputIndex2 = 2 ;
    const int outputIndex3 = 3 ;
    assert(engine.getBindingDataType(outputIndex1) == nvinfer1::DataType::kFLOAT);
    int mBatchSize =1;// engine.getMaxBatchSize();

    // Create GPU buffers on device
    
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], output_size1*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], output_size2*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex3], output_size3*sizeof(float)));

    // CHECK(cudaMalloc(&buffers[outputIndex2], output_size2*sizeof(float)));
    // CHECK(cudaMalloc(&buffers[outputIndex3], output_size3*sizeof(float)));
    // CHECK(cudaMalloc(&buffers[outputIndex4], output_size4*sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    // int32_t batchSize, void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed)
    // context.enqueue(1, buffers, stream, nullptr);
    // void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed)
    context.enqueueV2(buffers, stream, nullptr);
    
    CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], output_size1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output2, buffers[outputIndex2], output_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output3, buffers[outputIndex3], output_size3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // cout<<"IM here\n";
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    
    CHECK(cudaFree(buffers[inputIndex]));

    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
    CHECK(cudaFree(buffers[outputIndex3]));
    
}

/*
void parseOnnxModel(const string & model_path,
                    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    Logger logger;
    // first we create builder 
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    // then define flag that is needed for creating network definitiopn 
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    // then parse network 
    unique_ptr<nvonnxparser::IParser,TRTDestroy> parser{nvonnxparser::createParser(*network,logger)};
    // parse from file
    parser->parseFromFile(model_path.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    // lets create config file for engine 
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1U<<24);
    // config->setMaxWorkspaceSize(1U<<30);

    // use fp16 if it is possible 

    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // setm max bach size as it is very importannt for trt
    // builder->setMaxBatchSize(1);
    
    // create engine and excution context
    // unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    engine.reset(builder->buildEngineWithConfig(*network,*config));
    context.reset(engine->createExecutionContext());

    return;
}


*/
int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "run 'python3 yolox/deploy/trt.py -n yolox-{tiny, s, m, l, x}' to serialize model first!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./yolox ../model_trt.engine -i ../../../assets/dog.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }
    const std::string input_image_path {argv[3]};
    cv::VideoCapture cap(input_image_path);
    //std::vector<std::string> file_names;
    //if (read_files_in_dir(argv[2], file_names) < 0) {
        //std::cout << "read_files_in_dir failed." << std::endl;
        //return -1;
    //}
    
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    // unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine{nullptr};
    // unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context{nullptr};
    // parseOnnxModel(argv[1],engine,context);


    ///---------------------------------------------------------------------------

    delete[] trtModelStream;
    

    auto out_dims1 = engine->getBindingDimensions(1);
    // cout << engine->getBindingName(1)<<endl;
    auto output_size1 = 1;
    for(int j=0;j<out_dims1.nbDims;j++) {
        // std::cout<<"j "<<j<<"out_dims.d[j] "<<out_dims1.d[j]<<endl;
        output_size1 *= out_dims1.d[j];
    }

    // cout << "output_size: "<<output_size1<<endl;
    // cout <<"----------------------------\n";
    auto out_dims2 = engine->getBindingDimensions(2);
    // cout << engine->getBindingName(2)<<endl;
    auto output_size2 = 1;
    for(int j=0;j<out_dims2.nbDims;j++) {
        // std::cout<<"j "<<j<<"out_dims.d[j] "<<out_dims2.d[j]<<endl;
        output_size2 *= out_dims2.d[j];
    }

    // cout << "output_size: "<<output_size2<<endl;
    // cout <<"----------------------------\n";
    auto out_dims3 = engine->getBindingDimensions(3);
    // cout << engine->getBindingName(3)<<endl;
    auto output_size3 = 1;
    for(int j=0;j<out_dims3.nbDims;j++) {
        // std::cout<<"j "<<j<<"out_dims.d[j] "<<out_dims3.d[j]<<endl;
        output_size3 *= out_dims3.d[j];
    }

    // cout << "output_size: "<<output_size3<<endl;
    // cout <<"----------------------------\n";

    float* prob1 = new float[output_size1];
    float* prob2 = new float[output_size2];
    float* prob3 = new float[output_size3];
    cv::Mat img;
    float* blob;
    for(;;)
    {
        // cv::Mat img = cv::imread(input_image_path);
        cap >> img;

        // stop the program if no more images
        if(img.rows==0 || img.cols==0)
            break;
        int img_w = img.cols;
        int img_h = img.rows;
        cv::Mat pr_img = static_resize(img);
        // std::cout << "blob image" << std::endl;

        // float* blob;
        blob = blobFromImage(pr_img);
        float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        // cout<< pr_img.size()<<endl;
        // run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, blob, prob1,output_size1, prob2,output_size2, prob3,output_size3, pr_img.size());
        

        // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "ms" << std::endl;
        
        std::vector<Object> objects;
        decode_outputs(prob1,prob2,prob3, objects, scale, img_w, img_h);
        auto end = std::chrono::system_clock::now();
        auto micro= std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout<<1e6/micro<<" FPS"<<std::endl;
        draw_objects(img, objects, input_image_path);
        
        // break;
    }
    // delete the pointer to the float
    delete blob;
    // destroy the engine
    delete context;
    delete engine;
    delete runtime;
    //     context->destroy();
    // engine->destroy();
    // runtime->destroy();
    return 0;
}
