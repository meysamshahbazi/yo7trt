#include "yolo7.hpp"

#define CLASSIC_MEM_
#define UNIFIED_MEM_
Yolo7::Yolo7(std::string engine_file_path,int img_w ,int img_h)
{
    // TODO: check for existance of engine file or crete engine...
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    unique_ptr<IRuntime,TRTDestroy> runtime{createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine{runtime->deserializeCudaEngine(trtModelStream, size)};
    assert(engine != nullptr); 
    context.reset( engine->createExecutionContext() );
    assert(context != nullptr);
    
    delete[] trtModelStream;
    
    // cout<<"------------------------------"<<endl;
    // for (size_t i = 0; i < engine->getNbBindings(); ++i)
    // {
    //     auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * 1 * sizeof(float);
    //     // cudaMalloc(&buffers_base_q[i], binding_size);
    //     std::cout<<engine->getBindingName(i)<<std::endl;
    // }

    assert(engine->getNbBindings() == 4); // it must be 4
    auto out_dims1 = engine->getBindingDimensions(1);
    auto out_dims2 = engine->getBindingDimensions(2);
    auto out_dims3 = engine->getBindingDimensions(3);

    auto output_size1 = getSizeByDim(out_dims1);
    auto output_size2 = getSizeByDim(out_dims2);
    auto output_size3 = getSizeByDim(out_dims3);

#ifdef CLASSIC_MEM_
    prob1 = new float[output_size1];
    prob2 = new float[output_size2];
    prob3 = new float[output_size3];

    blob = new float[INPUT_H*INPUT_W*3];

    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_W * INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], output_size1*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], output_size2*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex3], output_size3*sizeof(float)));
#endif
#ifdef UNIFIED_MEM_
    cudaMallocManaged((void **)&blob,3 * INPUT_W * INPUT_H * sizeof(float),cudaMemAttachHost);
    cudaMallocManaged((void **)&prob1,output_size1*sizeof(float));
    cudaMallocManaged((void **)&prob2,output_size2*sizeof(float));
    cudaMallocManaged((void **)&prob3,output_size3*sizeof(float));

    buffers[inputIndex] = (void *) blob;
    buffers[outputIndex1] = (void *) prob1;
    buffers[outputIndex2] = (void *) prob2;
    buffers[outputIndex3] = (void *) prob3;
#endif
    
    CHECK(cudaStreamCreate(&stream));

}




