# yo7trt

## How to run:

- first clone my fork on yolov7 and generate trt engine:
```
git clone git@github.com:meysamshahbazi/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
python gen_engine.py

```
- then clone this repo and run these!

```
git clone git@github.com:meysamshahbazi/yo7trt.git
cd yo7trt
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make 
./yo7trt /path/to/genereted/engine -i /path/to/video/file

```

## ToDo:

- [x]  Add how to section

- [x]  Add sctript for generating trt model 

- [x]  Add section for generating onnx model in python

- [x]  Add function for parse and generate onnx model

- [ ] Clean code and modularization!

- [ ] use  `getBindingIndex()` for general coding

