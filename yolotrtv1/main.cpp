#include <iostream>
#include <string>
#include "utils/yolov11.h"
#include "../ops.h"

vector<Detection> detect(YOLOv11& model, Mat& image)
{
    vector<Detection> objects;
    
    model.preprocess(image);
    model.infer();
    model.postprocess(objects);
    model.draw(image, objects);
    imwrite("ret.jpg", image);
    return objects;
}

int main(int argc, char** argv)
{
    const string engine_file_path{ argv[1] };
    const string path{ argv[2] };
    assert(argc == 3);

    YOLOv11 model(engine_file_path);

    Mat image = imread(path);
    detect(model, image);

    // Mat image2 = imread(path);
    // detect(model, image2);

    // Mat image3 = imread(path);
    // detect(model, image3);
    // measure_exec_time(detect, model, image);

    return 0;
}