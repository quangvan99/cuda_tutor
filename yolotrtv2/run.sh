trtexec --onnx=onnx_model/yolov8n.onnx --saveEngine=onnx_model/yolov8n.engine --fp16

nvcc run.cpp process.cu -o run \
    -I/opt/TensorRT-8.6.1.6/include \
    -L/opt/TensorRT-8.6.1.6/lib \
    -lnvinfer \
    `pkg-config --cflags --libs opencv4` \
    -diag-suppress=611 -Wno-deprecated-declarations

./run ../../weights/yolo11s.engine ../im/bus.jpg 
