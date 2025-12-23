# Alinx_DNN

基于 Xilinx DPU 的 YOLOv3 实时障碍物检测（Vitis-AI v1.2 / DNNDK / Alinx AXU2CGB）

本仓库实现并评估一套面向嵌入式边缘端的 FPGA-DPU 推理流水线：将卷积等密集计算下发至 DPU（PL），预处理与后处理保留在 ARM（PS）端，以便可控地分析量化误差、性能瓶颈与系统级能效。

目录（摘要）

- `tf_yolov3_detect_deploy/`：部署与推理代码、模型数据与评估脚本
- `vitis-ai_v1.2_dnndk/`：板端 DNNDK 运行时包与安装脚本

快速导航

- 使用示例：见 “快速运行” 一节
- 评估指标：见 “实验结果” 一节

## 项目亮点

- ARM（PS）+ DPU（PL）异构推理：DPU 负责主计算，ARM 负责预处理/后处理与评估
- 提供 mAP / Recall 的评估逻辑，可用于对比量化前后精度变化
- 在 Alinx AXU2CGB（Zynq UltraScale+ MPSoC）+ DPUCZDX8G 上验证 INT8 量化后的性能与能效表现

## 目录结构

```
Alinx_DNN/
├─ tf_yolov3_detect_deploy/
│  ├─ model_data/                # classes_detect.txt, yolo_anchors_detect.txt
│  ├─ modellib/                  # DPU 调用封装（C++ / Python）
│  ├─ tf_yolov3_voc_pic.py        # 单图像推理示例
│  ├─ tf_yolov3_voc_cam.py        # 摄像头/实时推理示例
│  ├─ tf_yolov3_voc_ac.py         # FPGA 推理与评估主入口
│  └─ eval_mAP_dpu.py             # mAP / Recall 计算
├─ vitis-ai_v1.2_dnndk/           # 板端运行时打包（安装脚本）
└─ vitis-ai_v1.2_dnndk.tar.gz
```

## 硬件与软件要求

硬件（板端）

- 开发板：Alinx AXU2CGB（Zynq UltraScale+ MPSoC）
- 加速单元：DPUCZDX8G

软件（板端）

- OS：Embedded Linux（Petalinux SD 镜像）
- Runtime：Vitis-AI v1.2 / DNNDK（n2cube 等）
- Python：3.6/3.7（建议与 DNNDK 环境匹配）
- 依赖：numpy、opencv-python（或板端 opencv 运行库）

软件（主机端，用于模型准备/量化/编译，非必须）

- Ubuntu 18.04/20.04
- Vitis-AI v1.2（量化/编译）
- TensorFlow/Keras（训练或导出）

说明：本仓库重点是“板端部署与评估”。训练/量化/编译流程可独立复现，但不强制依赖主机环境。

## 快速运行

板端 DNNDK 安装（在板上执行）：

```bash
cd vitis-ai_v1.2_dnndk
chmod +x install.sh
sudo ./install.sh
n2cube --version   # 验证安装
```

示例：

```bash
# 单张图片推理（CPU/示例）
python3 tf_yolov3_detect_deploy/tf_yolov3_voc_pic.py

# 摄像头实时推理
python3 tf_yolov3_detect_deploy/tf_yolov3_voc_cam.py

# 在板端使用 DPU 模型进行推理与评估
python3 tf_yolov3_detect_deploy/tf_yolov3_voc_ac.py
```

## 推理流程概览

1. ARM（PS）端：预处理（resize / letterbox / normalize）
2. DPU（PL）端：执行 YOLOv3 主干网络与检测头的卷积计算
3. ARM（PS）端：后处理（输出解码、IoU 计算、NMS）
4. 输出检测结果；在评估模式下统计预测并计算 mAP / Recall

将预处理/后处理保留在 ARM 侧的目的：

- 便于对齐训练前处理，减少“部署不一致”导致的精度偏差
- 便于拆分统计各阶段耗时，定位系统瓶颈
- 便于评估量化误差与后处理阈值对指标的影响

## 实验结果

精度（mAP@0.5）

| 平台        | mAP@0.5 |
| ----------- | ------: |
| CPU         |  0.6103 |
| GPU         |  0.5857 |
| FPGA (INT8) |  0.4929 |

性能与功耗

| 平台 |   FPS | 单帧延迟 (ms) | 功耗 (W) |
| ---: | ----: | ------------: | -------: |
|  CPU |  2.82 |         354.8 |    22.61 |
|  GPU | 32.29 |         30.97 |    66.86 |
| FPGA | 17.18 |         58.21 |     8.87 |

能效（FPS/W）

| 平台 | FPS/W |
| ---: | ----: |
|  CPU | 0.125 |
|  GPU | 0.483 |
| FPGA | 1.937 |

讨论：在相同评估条件下 FPGA 显著优于 CPU/GPU 的能效表现，但 INT8 量化会带来精度下降；可通过量化感知训练（QAT）或更细的校准样本集减小损失。

## 可复现性说明

- 预处理逻辑与训练保持一致；Anchor 与类别文件位于 `tf_yolov3_detect_deploy/model_data/`。
- `tf_yolov3_detect_deploy/tf_yolov3_voc_ac.py` 为 FPGA 推理与评估主入口，`eval_mAP_dpu.py` 负责 mAP/Recall 计算。
- 所需工具链：Vitis-AI v1.2、Petalinux。训练/量化/编译步骤在本仓库中与部署流程相互独立，可按需复现。

## 开发建议与后续工作

- 若需提升 FPGA 精度：尝试量化感知训练（QAT）或更细的校准样本集。
- 若需提升吞吐：评估不同 DPU 配置或模型剪枝/轻量化策略。

## 许可证

本项目采用 MIT License，欢迎用于学习与研究。

## 致谢

- Xilinx / AMD Vitis-AI 团队
- Alinx FPGA 平台
- 课题组与指导老师

---
