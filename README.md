Deploy-YOLOv8-multitask
===
本仓库集成了YOLOv8的Detectation、Segmentation、Pose功能，代码主体通过C++和CUDA实现，通过TensorRT实现模型推理加速。


<table>
    <tr>
        <td><img src="data/result/car-detect-fp32.png" width="100%"></td>
        <td><img src="data/result/car-segment-fp32.png" width="100%"></td>
        <td><img src="data/result/car-pose-fp32.png" width="100%"></td>
    </tr>
</table>

代码已经在Ubuntu 20.04，CUDA 11.3，TensorRT 8.4.1上测试通过。

## Setup
1、安装CUDA、CUDNN、TensorRT，确保必要的C++和CUDA运行环境

2、创建一个虚拟环境并安装pytorch等必要python包
```
conda create -n yolov8-trt python=3.8
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
3、源码安装ultralytics
```
git clone https://github.com/xiaolu-luu/YOLOv8-TensorRT.git
cd ultralytics
# Install the package in editable mode for development
pip install -e .
```

## Run

1、导出ONNX模型文件（可跳过）
```
cd ultralytics/workspace
python export.py
```

2、修改Makefile 
```
# 根据当前的环境修改gcc和cuda的版本
CXX                         :=  g++
CUDA_VER                    :=  11.3

# opencv和TensorRT的安装目录
OPENCV_INSTALL_DIR          :=  /usr/include/opencv4
TENSORRT_INSTALL_DIR        :=  /home/ztl/tensorrt_learning/TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.1.5
```

3、编译运行

```
#编译
make
#运行
make run
```

## Notes

1、检查模型导出时输入输出的形状



<table style="width: 80%; margin: 0 auto;">
    <tr>
        <th>Task</th>
        <th>Detection</th>
        <th>Segmentation</th>
        <th>Pose</th>
    </tr>
    <tr>
        <td>Input</td>
        <td>1x3x640x640</td>
        <td>1x3x640x640</td>
        <td>1x3x640x640</td>
    </tr>
    <tr>
        <td>Output</td>
        <td>1x8400x84</td>
        <td>1x8400x116<br>1x32x160x160</td>
        <td>1x8400x56</td>
    </tr>
</table>