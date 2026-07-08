# Docker 环境搭建与运行流程

## 前置条件

- 服务器已安装 Docker 和 nvidia-container-toolkit
- 项目代码与 nuScenes 数据集在同一目录下
- 确认 GPU 可用：docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi

---

## 1. 构建 Docker 镜像（约 10-15 分钟，只需一次）

Dockerfile 位于项目 docker/ 子目录，基于 CUDA 11.3 + Ubuntu 20.04，安装了 Python 3.8、PyTorch 1.10.1、MMCV 1.4.0 等依赖。

```bash
cd /path/to/Semantic-BEVFusion/docker
docker build . -t bevfusion
```

| 参数 | 含义 |
|------|------|
| . | 构建上下文为当前目录 |
| -t bevfusion | 镜像名称（tag），后续通过此名称启动容器 |

> 如果 pip 下载超时（国内服务器常见），在 Dockerfile 中加 ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple。

---

## 2. 启动容器

```bash
docker run --gpus all -it --rm \
  -v /data/data3/share_data/lixiaoxiao19/code/Semantic-BEVFusion:/workspace \
  --shm-size 16g \
  -w /workspace \
  bevfusion /bin/bash
```

| 参数 | 含义 |
|------|------|
| --gpus all | 将所有 GPU 分配给容器 |
| -it | 交互模式 + 伪终端 |
| --rm | 退出后自动删除容器 |
| -v 宿主机:容器 | 挂载目录，容器内修改直接反映到宿主机 |
| --shm-size 16g | 共享内存大小，PyTorch DataLoader 多进程加载数据必需 |
| -w /workspace | 容器启动后默认工作目录 |
| bevfusion | 镜像名称，对应构建时的 -t |

---

## 3. 容器内初始化（首次启动必做）

```bash
# 编译 CUDA 扩展（spconv、bev_pool 等自定义算子，约 5-10 分钟）
python setup.py develop

# 修复几个兼容性问题（只做一次）
pip uninstall opencv-python -y
pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install yapf==0.40.1 setuptools==59.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4. 运行测试


```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --workers 10 --version v1.0-mini

```

```bash
CUDA_VISIBLE_DEVICES=2 torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth \
  --eval map \
  --out work_dirs/bevfusion-seg/results.pkl
```

| 参数 | 含义 |
|------|------|
| CUDA_VISIBLE_DEVICES=2 | 指定使用编号为 2 的 GPU |
| 	orchpack dist-run | 分布式任务启动器 |
| -np 1 | 使用 1 个进程（单卡），多卡改为 -np 8 |
| 	ools/test.py | 测试入口脚本 |
| 第一个参数 | YAML 配置文件路径，定义模型结构、数据集、pipeline |
| 第二个参数 | 预训练权重 .pth 路径 |
| --eval map | 评测指标类型（分割：map，检测：box） |


```bash
CUDA_VISIBLE_DEVICES=2 torchpack dist-run -np 1 python tools/visualize.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  --mode pred \
  --checkpoint pretrained/bevfusion-seg.pth \
  --split val \
  --out-dir results/bevfusion-seg/viz \
  --map-score 0.5

```

### 测试命令速查

```bash
# BEV 地图分割
CUDA_VISIBLE_DEVICES=2 torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth \
  --eval map

# 3D 目标检测
CUDA_VISIBLE_DEVICES=2 torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox
```

---

## 5. 运行训练

```bash
CUDA_VISIBLE_DEVICES=2 torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  --data.samples_per_gpu 1 \
  --data.workers_per_gpu 2
```

| 参数 | 含义 |
|------|------|
| 	ools/train.py | 训练入口脚本 |
| YAML 配置路径 | 模型结构、优化器、数据 pipeline 等 |
| --model.encoders.camera.backbone.init_cfg.checkpoint | 覆盖 Camera backbone 的预训练权重路径 |
| --data.samples_per_gpu 1 | 每 GPU batch size，单卡 OOM 时降低此值 |
| --data.workers_per_gpu 2 | 每 GPU 数据加载进程数 |

训练过程中关键日志指标：

| 指标 | 含义 |
|------|------|
| lr | 当前学习率 |
| loss | 总损失 |
| loss/map/xxx/focal | 各类别 focal loss |
| grad_norm | 梯度范数（用于监控梯度爆炸） |
| memory | 当前 GPU 显存占用（MB） |
| eta | 预计剩余时间 |

---

## 6. 常见问题速查

| 现象 | 原因 | 解决 |
|------|------|------|
| cuda execution failed with error 2 | 显存不足 | 换空闲 GPU 或降低 samples_per_gpu |
| 
vcc fatal: Unsupported gpu architecture 'compute_89' | CUDA 11.3 不支持 sm_89 | 删除 setup.py 中 compute_89 那行；L40 会通过 PTX JIT 运行 |
| ImportError: libQt5Core... | opencv 缺 Qt5 库 | 用 opencv-python-headless 替代 |
| EOFError: Ran out of input | 预训练权重为空（git-lfs 指针） | 从本地 scp 真实文件到服务器 |
| DataLoader 卡死 | 共享内存不足 | 启动容器时确认 --shm-size 16g |

---

## 7. BEV 语义图颜色含义

使用以下命令生成预测 BEV 地图：

```bash
CUDA_VISIBLE_DEVICES=2 torchpack dist-run -np 1 python tools/visualize.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  --mode pred \
  --checkpoint pretrained/bevfusion-seg.pth \
  --split val \
  --out-dir results/bevfusion-seg/viz \
  --map-score 0.5
```

输出图片位于：

```text
results/bevfusion-seg/viz/map/*.png
```

| 颜色 | RGB | 类别 | 含义 |
|------|-----|------|------|
| 浅蓝色 | `(166, 206, 227)` | `drivable_area` | 可行驶区域，道路车行区域 |
| 浅红/粉色 | `(251, 154, 153)` | `ped_crossing` | 人行横道 |
| 红色 | `(227, 26, 28)` | `walkway` | 人行道、步行区域 |
| 浅橙色 | `(253, 191, 111)` | `stop_line` | 停止线 |
| 橙色 | `(255, 127, 0)` | `carpark_area` | 停车区域 |
| 紫色 | `(106, 61, 154)` | `divider` | 道路/车道分隔线，合并 `road_divider` 和 `lane_divider` |
| 浅灰色 | `(240, 240, 240)` | background | 未预测为上述类别的背景区域 |

说明：

- `--mode pred` 输出的是模型预测结果，不是 GT。
- `--map-score 0.5` 表示每个语义通道概率 `>= 0.5` 才会被画出来。
- `divider` 不区分 `road_divider` 和 `lane_divider`，两者都会显示为同一种紫色。
- 多个 mask 在同一像素重叠时，`visualize_map()` 按 `map_classes` 顺序上色，后面的类别会覆盖前面的类别；当前 `divider` 最后绘制，因此紫色分隔线会覆盖其它类别。


