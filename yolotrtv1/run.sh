trtexec --onnx=onnx_model/yolov8n.onnx --saveEngine=onnx_model/yolov8n.engine --fp16
nvcc run.cpp utils/yolov11.cpp utils/preprocess.cu -o run  \
    -I/opt/TensorRT-8.6.1.6/include  \
    -L/opt/TensorRT-8.6.1.6/lib  \
    -lnvinfer    \
    `pkg-config --cflags --libs opencv4` -diag-suppress=611 -Wno-deprecated-declarations


nvcc -c preprocess.cu -o preprocess.o `pkg-config --cflags --libs opencv4`
g++ run.cpp utils/yolov11.cpp utils/preprocess.o -o run \
    -I/opt/TensorRT-8.6.1.6/include \
    -I/usr/local/cuda/include \
    -L/opt/TensorRT-8.6.1.6/lib \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart \
    `pkg-config --cflags --libs opencv4` -diag-suppress=611 -Wno-deprecated-declarations

./run ../../weights/yolo11s.engine ../im/bus.jpg 


nvcc -c postprocess.cu -o postprocess.o -I/opt/TensorRT-8.6.1.6/include `pkg-config --cflags --libs opencv4`