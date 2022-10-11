#include "utils.hpp"
#include "yolo7.hpp"

#define CLASSIC_MEM
// #define UNIFIED_MEM

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.5

/*
static Logger gLogger;

void generete_proposal_scale(float* feat_blob, float prob_threshold, std::vector<Object>& objects,int nx, int ny,float stride,float * anchor_grid_w,float * anchor_grid_h)
{
    for(int i1=0;i1<3;i1++)
    for(int i2=0;i2<ny;i2++) // nx = 80 stride = 8 
    for(int i3=0;i3<nx;i3++) // ny = 48
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
        
        if (obj_conf > prob_threshold)
        {
            std::array<float, NUM_CLASSES> &classes_scores = reinterpret_cast<std::array<float, NUM_CLASSES>&>(feat_blob[i4+this_index]);

            auto max_score = std::max_element(classes_scores.begin() , classes_scores.end()); 
            int argmaxVal = distance(classes_scores.begin(), max_score);

            float prob = classes_scores[argmaxVal]*obj_conf;

            Object obj;
            obj.prob = prob;
            obj.label = argmaxVal;
            float x0 = x - w * 0.5f;
            float y0 = y - h * 0.5f;
            obj.rect = cv::Rect_<float>(x0,y0,w,h);
            objects.emplace_back(obj);
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

}

// float* blobFromImage(cv::Mat& img){

void blobFromImage(cv::Mat& img,float* blob){
    int img_h = img.rows;
    int img_w = img.cols;
    int data_idx = 0;
    for (int i = 0; i < img_h; ++i)
    {
        uchar* pixel = img.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img_w; ++j)
        {
            blob[data_idx] = (*pixel++)/255.f;
            blob[data_idx+img_h*img_w] = (*pixel++)/255.f;
            blob[data_idx+2*img_h*img_w] = (*pixel++)/255.f;
            data_idx++;
        }
    }
}

void blobFromImage2(cv::Mat& img,float* blob){
    // this function optimized for padded image copy!
    int img_h = img.rows;
    int img_w = img.cols;
    int data_idx = 0;
    for (int i = 0; i < img_h; ++i)
    {
        uchar* pixel = img.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img_w; ++j)
        {
            data_idx = i*img_w+j;
            blob[data_idx] = (*pixel++)/255.f;
            blob[data_idx+INPUT_H*INPUT_W] = (*pixel++)/255.f;
            blob[data_idx+2*INPUT_H*INPUT_W] = (*pixel++)/255.f;
            // data_idx++;
        }
    }
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
*/
int main(int argc, char** argv) 
{
    const std::string input_image_path {argv[3]};
    cv::VideoCapture cap(input_image_path);
    cv::Mat img;
    cap >> img;

    // stop the program if no more images
    if(img.rows==0 || img.cols==0)
        return -1;
    int img_w = img.cols;
    int img_h = img.rows;
    Yolo7 yolo7(argv[1],img_w,img_h);

    // cout<<"im here"
    for(;;)
    {
        cap >> img;
        // stop the program if no more images
        if(img.rows==0 || img.cols==0)
            break;
        
        std::vector<Object> objects;
        yolo7.detect(img,objects);
        
        draw_objects(img, objects, input_image_path);
        
        int key = cv::waitKey(1); 
            if(key == 'q') break;

    }
    return 0;
}


/*
int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    // saveEngineFile(argv[1],"../yolo-tiny.engine");
    // return -1;

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
    
    // IRuntime* runtime = createInferRuntime(gLogger);
    unique_ptr<IRuntime,TRTDestroy> runtime{createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    // ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine{runtime->deserializeCudaEngine(trtModelStream, size)};
    assert(engine != nullptr); 
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context{engine->createExecutionContext()};
    // IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    
    delete[] trtModelStream;
    
    

    // parseOnnxModel(argv[1],1U<<30,engine,context);
    // unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine{nullptr};
    // unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context{nullptr};
    
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * 1 * sizeof(float);
        // cudaMalloc(&buffers_base_q[i], binding_size);
        std::cout<<engine->getBindingName(i)<<std::endl;
    }

    cv::Mat img;

    // code from doInference()
    assert(engine->getNbBindings() == 4); // it must be 4
    void* buffers[4];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = 0;//= engine->getBindingIndex(INPUT_BLOB_NAME);

    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex1 = 1;
    const int outputIndex2 = 2;
    const int outputIndex3 = 3;
    assert(engine->getBindingDataType(outputIndex1) == nvinfer1::DataType::kFLOAT);
    int mBatchSize =1;// engine.getMaxBatchSize();

    // Create GPU buffers on device

    auto out_dims1 = engine->getBindingDimensions(1);
    auto out_dims2 = engine->getBindingDimensions(2);
    auto out_dims3 = engine->getBindingDimensions(3);

    auto output_size1 = getSizeByDim(out_dims1);
    auto output_size2 = getSizeByDim(out_dims2);
    auto output_size3 = getSizeByDim(out_dims3);

#ifdef CLASSIC_MEM
    float* prob1 = new float[output_size1];
    float* prob2 = new float[output_size2];
    float* prob3 = new float[output_size3];

    float blob[INPUT_H*INPUT_W*3];

    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_W * INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], output_size1*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], output_size2*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex3], output_size3*sizeof(float)));
#endif
#ifdef UNIFIED_MEM
    float *blob;
    float *prob1,*prob2,*prob3;
    cudaMallocManaged((void **)&blob,3 * INPUT_W * INPUT_H * sizeof(float),cudaMemAttachHost);
    cudaMallocManaged((void **)&prob1,output_size1*sizeof(float));
    cudaMallocManaged((void **)&prob2,output_size2*sizeof(float));
    cudaMallocManaged((void **)&prob3,output_size3*sizeof(float));

    buffers[inputIndex] = (void *) blob;
    buffers[outputIndex1] = (void *) prob1;
    buffers[outputIndex2] = (void *) prob2;
    buffers[outputIndex3] = (void *) prob3;
#endif
    cudaStream_t stream;
    
    CHECK(cudaStreamCreate(&stream));
    // cv::Mat pr_img;

    cap >> img;

        // stop the program if no more images
    if(img.rows==0 || img.cols==0)
        return -1;
    int img_w = img.cols;
    int img_h = img.rows;

    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));    
    int unpad_w = scale * img_w;
    int unpad_h = scale * img_h;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);

    int data_idx = 0;
    for (int i = 0; i < INPUT_H; ++i)
    {
        // uchar* pixel = img.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < INPUT_W; ++j)
        {
            blob[data_idx] = 114.0f/255.f;
            blob[data_idx+INPUT_H*INPUT_W] = 114.0f/255.f;
            blob[data_idx+2*INPUT_H*INPUT_W] = 114.0f/255.f;
            data_idx++;
        }
    }


    for(;;)
    {
        cap >> img;
        // stop the program if no more images
        if(img.rows==0 || img.cols==0)
            break;

        int img_w = img.cols;
        int img_h = img.rows;
        cout<<img_w<<" "<<img_h<<endl;
        cv::resize(img, re, re.size());
        blobFromImage2(re,blob);

        auto start = std::chrono::system_clock::now();
#ifdef CLASSIC_MEM
        CHECK(cudaMemcpyAsync(buffers[inputIndex], blob, 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream));
#endif

#ifdef UNIFIED_MEM
        cudaStreamAttachMemAsync(stream,blob,0,cudaMemAttachGlobal);
#endif
         // int32_t batchSize, void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed)
        // context.enqueue(1, buffers, stream, nullptr);
        // void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed)
        context->enqueueV2(buffers, stream, nullptr);
#ifdef CLASSIC_MEM       
        CHECK(cudaMemcpyAsync(prob1, buffers[outputIndex1], output_size1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(prob2, buffers[outputIndex2], output_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(prob3, buffers[outputIndex3], output_size3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
#endif        
        // cout<<"IM here\n";
#ifdef UNIFIED_MEM
        cudaStreamSynchronize(stream); // this additional bcuse of Note on page 10 cuda for tegra doc 
        cudaStreamAttachMemAsync(stream,prob1,0,cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream,prob2,0,cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream,prob3,0,cudaMemAttachHost);
#endif
        cudaStreamSynchronize(stream);
         
        std::vector<Object> objects;
        decode_outputs(prob1,prob2,prob3, objects, scale, img_w, img_h);
        auto end = std::chrono::system_clock::now();
        auto micro= std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout<<1e6/micro<<" FPS"<<std::endl;
        draw_objects(img, objects, input_image_path);
        
        int key = cv::waitKey(1); 
            if(key == 'q') break;

#ifdef UNIFIED_MEM
        cudaStreamAttachMemAsync(stream,blob,0,cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream,prob1,0,cudaMemAttachGlobal);
        cudaStreamAttachMemAsync(stream,prob2,0,cudaMemAttachGlobal);
        cudaStreamAttachMemAsync(stream,prob3,0,cudaMemAttachGlobal);
#endif

    }

    cudaStreamDestroy(stream);
    
    CHECK(cudaFree(buffers[inputIndex]));

    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
    CHECK(cudaFree(buffers[outputIndex3]));


    // delete the pointer to the float
    // delete[] blob;
    // destroy the engine
    // delete context;
    // delete engine;
    // delete runtime;
    return 0;
}

*/