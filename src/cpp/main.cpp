#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    /*这么实现目的在于让调用的整个过程精简化*/
    string onnxPath    = "models/onnx/yolov8n-pose.onnx";

    auto level         = logger::Level::VERB;
    auto params        = model::Params();

    params.img         = {640, 640, 3};
    params.task        = model::task_type::POSETASK;
    params.dev         = model::device::GPU;
    params.prec        = model::precision::FP32;

    // 创建一个worker的实例, 在创建的时候就完成初始化
    auto worker   = thread::create_worker(onnxPath, level, params);

    // 根据worker中的task类型进行推理
    worker->inference("data/source/car.jpg");
    worker->inference("data/source/airport.jpg");
    worker->inference("data/source/crossroad.jpg");
    worker->inference("data/source/bedroom.jpg");

    return 0;
}

