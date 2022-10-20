#include "utils.hpp"
#include "yolo7.hpp"

#define CLASSIC_MEM
// #define UNIFIED_MEM

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.5


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
        auto start = std::chrono::system_clock::now();
        yolo7.detect(img,objects);

        auto end = std::chrono::system_clock::now();
        auto micro= std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout<<1e6/micro<<" FPS"<<std::endl;
        draw_objects(img, objects, input_image_path);
        
        int key = cv::waitKey(1); 
            if(key == 'q') break;

    }
    return 0;
}
