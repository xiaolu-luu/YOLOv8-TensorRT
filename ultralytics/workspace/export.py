from ultralytics import YOLO
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Load a model
model_detect = YOLO("yolov8n.pt")
model_segmentation = YOLO("yolov8n-seg.pt")
model_pose = YOLO("yolov8n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model_detect.export(format="onnx")
model_segmentation.export(format="onnx", opset=12, dynamic=False, imgsz=640)
model_pose.export(format="onnx", opset=12, dynamic=False, imgsz=640)

graph = gs.import_onnx(onnx.load("yolov8n-seg.onnx")) 

# 找到模型的输出节点
output_tensors = graph.outputs

# 创建Transpose节点，这里假设我们对第一个输出节点进行Transpose操作
transpose_node = gs.Node(
    op ="Transpose",
    name="transpose_output",
    inputs=[output_tensors[0]],
    outputs=[gs.Variable(name="transposed_output", dtype=output_tensors[0].dtype, shape=[1,8400,116])],
    # attributes=[gs.Attribu(name="perm", values=np.array([1, 2], dtype=np.int64))]
    attrs   = {"perm":[0,2,1]}
)

# 将Transpose节点添加到图中
graph.nodes.append(transpose_node)

# 更新模型的输出为Transpose节点的输出
graph.outputs = [transpose_node.outputs[0],output_tensors[1]]

# 清理和拓扑排序图
graph.cleanup().toposort()

# 导出修改后的ONNX模型
onnx.save(gs.export_onnx(graph), "yolov8n-seg.onnx")