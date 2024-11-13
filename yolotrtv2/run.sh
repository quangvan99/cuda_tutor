nvcc main.cpp process.cu -o main \
    -I/opt/TensorRT-8.6.1.6/include \
    -L/opt/TensorRT-8.6.1.6/lib \
    -lnvinfer \
    `pkg-config --cflags --libs opencv4` \
    -diag-suppress=611 -Wno-deprecated-declarations

./main ../../weights/yolo11s.engine ../im/bus.jpg 

rm -rf main

# trtexec --onnx=onnx_model/yolov8n.onnx --saveEngine=onnx_model/yolov8n.engine --fp16