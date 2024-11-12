#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include "../ops.h"

using namespace nvinfer1;

#define MAX_OBJECTS 500
#define NUM_BOX_ELEMENT 7
#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000

#define CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 80;  // 80

const char* INPUT_BLOB_NAME = "images";    // onnx
const char* OUTPUT_BLOB_NAME = "output0";  // onnx

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;



struct affine_matrix
{
    float i2d[6];
    float d2i[6];
};

struct bbox
{
    float x1, y1, x2, y2;
    float score;
    int label;
};

// // Declaration of CUDA kernels (assumed to be implemented elsewhere)
void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           float* d2i, cudaStream_t stream);  // Letterbox CUDA

void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold,
    float* invert_affine_matrix, float* parray,
    int max_objects, cudaStream_t stream);  // CUDA

void nms_kernel_invoker(
    float* parray, float nms_threshold, int max_objects, cudaStream_t stream);  // NMS CUDA

void transpose_kernel_invoker(float* src, int num_bboxes, int num_elements, float* dst, cudaStream_t stream);

////////////////////////////////////////////////////////////
// Affine projection function
void affine_project(float* d2i, float x, float y, float* ox, float* oy)
{
    *ox = x * d2i[0] + y * d2i[1] + d2i[2];
    *oy = x * d2i[3] + y * d2i[4] + d2i[5];
}

// Calculate affine matrix and its inverse
void get_affine_martrix(affine_matrix& afmt, cv::Size& to, cv::Size& from)
{
    float scale = std::min(to.width / (float)from.width, to.height / (float)from.height);
    afmt.i2d[0] = scale;
    afmt.i2d[1] = 0;
    afmt.i2d[2] = (-scale * from.width + to.width) * 0.5;
    afmt.i2d[3] = 0;
    afmt.i2d[4] = scale;
    afmt.i2d[5] = (-scale * from.height + to.height) * 0.5;
    cv::Mat cv_i2d(2, 3, CV_32F, afmt.i2d);
    cv::Mat cv_d2i(2, 3, CV_32F, afmt.d2i);
    cv::invertAffineTransform(cv_i2d, cv_d2i);
    memcpy(afmt.d2i, cv_d2i.ptr<float>(0), sizeof(afmt.d2i));
}

class Detector
{
public:
    Detector(const std::string& engine_path);
    ~Detector();

    std::vector<bbox> detect(const cv::Mat& img);
    void draw(cv::Mat& img);

private:
    void loadModel(const std::string& engine_path);
    void prepareBuffers();
    void preprocess(const cv::Mat& img);
    void infer();
    void postprocess();

    // Members for inference
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;

    float* buffers[2];
    int inputIndex;
    int outputIndex;
    int OUTPUT_CANDIDATES;
    int output_size;

    float* affine_matrix_d2i_host;
    float* affine_matrix_d2i_device;

    float* decode_ptr_host;
    float* decode_ptr_device;
    float* transpose_device;

    cudaStream_t stream;
    uint8_t* img_host;
    uint8_t* img_device;

    // Members to hold input/output data
    affine_matrix afmt;

    // Results
    std::vector<bbox> boxes;
};


Detector::Detector(const std::string& engine_path)
{
    cudaSetDevice(DEVICE);
    loadModel(engine_path);
    prepareBuffers();
    CHECK(cudaStreamCreate(&stream));
}

Detector::~Detector()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
    cudaStreamDestroy(stream);
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(decode_ptr_device));
    CHECK(cudaFree(affine_matrix_d2i_device));
    CHECK(cudaFreeHost(affine_matrix_d2i_host));
    CHECK(cudaFree(transpose_device));

    delete[] decode_ptr_host;
}

void Detector::loadModel(const std::string& engine_path)
{
    // Read the engine file
    std::ifstream engineStream(engine_path, std::ios::binary);
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();
}

void Detector::prepareBuffers()
{
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Allocate GPU buffers
    CHECK(cudaMalloc((void**)&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));

    auto out_dims = engine->getBindingDimensions(outputIndex);
    output_size = 1;
    OUTPUT_CANDIDATES = out_dims.d[2];  // 8400

    for (int j = 0; j < out_dims.nbDims; j++)
    {
        output_size *= out_dims.d[j];
    }

    CHECK(cudaMalloc((void**)&buffers[outputIndex], output_size * sizeof(float)));
    CHECK(cudaMalloc(&transpose_device, output_size * sizeof(float)));

    // Allocate host and device buffers
    CHECK(cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6));
    CHECK(cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6));

    decode_ptr_host = new float[1 + MAX_OBJECTS * NUM_BOX_ELEMENT];
    CHECK(cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));

    // Prepare image buffers
    CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
}

void Detector::preprocess(const cv::Mat& img)
{
    cv::Size from(img.cols, img.rows);
    cv::Size to(INPUT_W, INPUT_H);
    get_affine_martrix(afmt, to, from);

    memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));
    CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));

    float* buffer_idx = buffers[inputIndex];
    size_t size_image = img.cols * img.rows * 3;
    memcpy(img_host, img.data, size_image);

    CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));

    // Preprocess the image
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, affine_matrix_d2i_device, stream);
}

void Detector::infer()
{
    context->enqueueV2((void**)buffers, stream, nullptr);
}

void Detector::postprocess()
{
    float* predict = buffers[outputIndex];
    transpose_kernel_invoker(predict, OUTPUT_CANDIDATES, NUM_CLASSES + 4, transpose_device, stream);

    predict = transpose_device;
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream));
    decode_kernel_invoker(predict, OUTPUT_CANDIDATES, NUM_CLASSES, BBOX_CONF_THRESH, affine_matrix_d2i_device, decode_ptr_device, MAX_OBJECTS, stream);
    nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream);

    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Parse results
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    for (int i = 0; i < count; i++)
    {
        int basic_pos = 1 + i * NUM_BOX_ELEMENT;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1)
        {
            bbox box;
            box.x1 = decode_ptr_host[basic_pos + 0];
            box.y1 = decode_ptr_host[basic_pos + 1];
            box.x2 = decode_ptr_host[basic_pos + 2];
            box.y2 = decode_ptr_host[basic_pos + 3];
            box.score = decode_ptr_host[basic_pos + 4];
            box.label = (int)decode_ptr_host[basic_pos + 5];
            boxes.push_back(box);
        }
    }
}

void Detector::draw(cv::Mat& img)
{
    for (const auto& box : boxes)
    {
        cv::Rect roi_area(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        cv::rectangle(img, roi_area, cv::Scalar(0, 255, 0), 2);
        std::string label_string = std::to_string(box.label) + " " + std::to_string(box.score);
        cv::putText(img, label_string, cv::Point(box.x1, box.y1 - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2);
    }
}

std::vector<bbox> Detector::detect(const cv::Mat& img)
{
    boxes.clear();

    preprocess(img);
    infer();
    postprocess();

    return boxes;
}

void detect(Detector& detector, cv::Mat& img){
    
    detector.detect(img);
    // detector.draw(img);
    // cv::imwrite("ret.jpg", img);
}

int main(int argc, char** argv)
{
    const std::string engine_path{ argv[1] };
    const std::string path{ argv[2] };
    assert(argc == 3);

    Detector detector(engine_path);
    cv::Mat img = cv::imread(path);
    // detect(detector, path);
    measure_exec_time(detect, detector, img);

    return 0;
}
