# Semantic-BEVFusion 代码结构快速理解

本文档面向快速上手和后续算法优化。建议先按“训练入口 -> 配置系统 -> 数据 pipeline -> 模型前向 -> loss/评估”的顺序读代码，不要一开始陷入 CUDA op 或所有 bbox 工具细节。

## 1. 项目一句话概括

该项目是一个基于 BEVFusion / MMDetection3D 风格的 3D 感知代码库，核心思路是把 camera、LiDAR、radar 等传感器特征统一到 BEV 空间，再做多模态融合，最后接 3D 检测头或 BEV 地图分割头。

当前代码相比原始 BEVFusion 主要多了两类改造：

- radar 数据链路和 `RadarEncoder`
- BEVDepth / AwareBEVDepth / DBEVDepth / AwareDBEVDepth 深度监督与深度感知 view transform

## 2. 顶层目录速查

```text
Semantic-BEVFusion/
├── README.md                         # 原项目说明、训练测试命令、依赖说明
├── setup.py                          # 安装项目并编译 CUDA/C++ 扩展
├── configs/                          # YAML 实验配置
├── tools/                            # 训练、测试、导出、可视化、数据转换入口
├── mmdet3d/                          # 核心代码
│   ├── apis/                         # train/test 高层 API
│   ├── core/                         # bbox、points、post-process、voxel 等基础结构
│   ├── datasets/                     # NuScenes dataset 和数据增强 pipeline
│   ├── models/                       # 模型、backbone、neck、head、fuser、view transform
│   ├── ops/                          # 自定义 CUDA/C++ 算子
│   ├── runner/                       # 自定义 runner
│   └── utils/                        # logger、config、syncbn 等工具
├── docker/                           # Docker 环境
├── assets/                           # README 资源
└── selffile/                         # 个人阅读笔记和项目文档
```

最重要的代码集中在：

- `tools/train.py`
- `tools/test.py`
- `configs/`
- `mmdet3d/datasets/`
- `mmdet3d/models/fusion_models/bevfusion.py`
- `mmdet3d/models/vtransforms/`
- `mmdet3d/models/backbones/radar_encoder.py`
- `mmdet3d/models/heads/`

## 3. 推荐阅读顺序

### 第一步：入口脚本

先读：

- `tools/train.py`
- `tools/test.py`

训练入口主流程：

```text
tools/train.py
  -> torchpack configs.load(...)
  -> recursive_eval(...) 解析 ${...}
  -> build_dataset(cfg.data.train)
  -> build_model(cfg.model)
  -> model.init_weights()
  -> train_model(...)
```

测试入口主流程：

```text
tools/test.py
  -> 加载配置
  -> build_dataset(cfg.data.test)
  -> build_model(cfg.model)
  -> load_checkpoint(...)
  -> multi_gpu_test(...)
  -> dataset.evaluate(...)
```

注意：`tools/test.py` 虽然保留了部分 OpenMMLab 的 launcher 参数，但当前代码基本按分布式执行路径写，通常用 `torchpack dist-run` 启动。

### 第二步：配置文件

先读：

- `configs/default.yaml`
- `configs/nuscenes/default.yaml`
- `configs/nuscenes/det/default.yaml`
- 你实际运行的具体 YAML，例如：
  - `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`
  - `configs/nuscenes/det/centerhead/lssfpn/camera+radar/resnet50/default.yaml`
  - `configs/nuscenes/det/centerhead/lssfpn/camera+radar/resnet50/dlss.yaml`
  - `configs/nuscenes/seg/fusion-bev256d2-lss.yaml`

本项目使用 `torchpack.utils.config.configs` 加载 YAML，并启用 `recursive=True`。理解配置时可以按路径逐级看 `default.yaml`，再看最终目标配置。配置中的 `${...}` 会被 `mmdet3d/utils/config.py` 里的 `recursive_eval` 解析。

例子：

```yaml
image_size: [256, 704]

model:
  encoders:
    camera:
      vtransform:
        image_size: ${image_size}
        feature_size: ${[image_size[0] // 16, image_size[1] // 16]}
```

常见配置块含义：

| 配置块 | 作用 |
|---|---|
| `model` | 定义 BEVFusion 结构，包括 encoders、fuser、decoder、heads |
| `data` | 定义 train/val/test dataset 和 pipeline |
| `train_pipeline` / `test_pipeline` | 数据加载、增强、格式化顺序 |
| `optimizer` | 优化器配置 |
| `lr_config` | 学习率策略 |
| `runner` | epoch/iter runner |
| `fp16` | 混合精度训练 |
| `evaluation` | 验证评估配置 |

## 4. 训练调用链

核心文件：

- `tools/train.py`
- `mmdet3d/apis/train.py`
- `mmdet3d/runner/epoch_based_runner.py`

完整链路：

```text
tools/train.py
  -> mmdet3d.apis.train_model
      -> build_dataloader
      -> MMDistributedDataParallel(model.cuda())
      -> build_optimizer
      -> build_runner
      -> register_training_hooks
      -> register eval hook
      -> resume/load checkpoint
      -> runner.run(...)
```

自定义 runner：

```python
class CustomEpochBasedRunner(EpochBasedRunner):
    def set_dataset(self, dataset):
        self._dataset = dataset

    def train(self, data_loader, **kwargs):
        for dataset in self._dataset:
            dataset.set_epoch(self.epoch)
        super().train(data_loader, **kwargs)
```

它的作用是在每个 epoch 开始前把当前 epoch 传给 dataset，主要服务于数据增强调度，例如 `ObjectPaste` 到某个 epoch 后停止。

## 5. 数据链路

核心文件：

- `mmdet3d/datasets/nuscenes_dataset.py`
- `mmdet3d/datasets/pipelines/loading.py`
- `mmdet3d/datasets/pipelines/transforms_3d.py`
- `mmdet3d/datasets/pipelines/formating.py`
- `tools/data_converter/nuscenes_converter.py`

### 5.1 Dataset 做什么

`NuScenesDataset` 继承自 `Custom3DDataset`，主要负责：

- 从 `ann_file` 加载 info pkl
- 根据 index 取出一帧样本信息
- 提供图片路径、点云路径、sweeps、标定矩阵、GT boxes、radar 信息
- 评估检测或分割结果

`get_data_info()` 输出的关键字段：

| 字段 | 含义 |
|---|---|
| `lidar_path` | 当前帧 LiDAR 点云 |
| `sweeps` | 历史 LiDAR sweeps |
| `image_paths` | 多相机图片路径 |
| `camera_intrinsics` | 相机内参 |
| `camera2ego` | camera 到 ego 坐标变换 |
| `lidar2ego` | LiDAR 到 ego 坐标变换 |
| `lidar2camera` | LiDAR 到 camera 坐标变换 |
| `lidar2image` | LiDAR 到 image 投影矩阵 |
| `camera2lidar` | camera 到 LiDAR 坐标变换 |
| `radar` | radar sweeps 信息，来自 info pkl 中的 `radars` |

### 5.2 Pipeline 顺序

检测任务的典型训练 pipeline：

```text
LoadMultiViewImageFromFiles
LoadPointsFromFile
LoadPointsFromMultiSweeps
LoadRadarPointsMultiSweeps
LoadAnnotations3D
ObjectPaste
ImageAug3D
GlobalRotScaleTrans
RandomFlip3D
PointsRangeFilter
ObjectRangeFilter
ObjectNameFilter
ImageNormalize
GridMask
PointShuffle
DefaultFormatBundle3D
Collect3D
GTDepth
```

其中：

- `LoadMultiViewImageFromFiles` 读取多相机图片
- `LoadPointsFromFile` 读取当前 LiDAR 点云
- `LoadPointsFromMultiSweeps` 读取历史 LiDAR sweeps 并变换到当前 LiDAR 坐标系
- `LoadRadarPointsMultiSweeps` 读取多 radar sweeps，并可补偿速度、过滤和归一化
- `ImageAug3D` 做图像 resize/crop/flip/rotate，并同步更新图像增强矩阵
- `GlobalRotScaleTrans` 和 `RandomFlip3D` 做 3D 增强，并同步处理 points、radar、boxes
- `GTDepth` 生成 camera view 下的深度监督
- `DefaultFormatBundle3D` 把 numpy / points / boxes 转成 MMDetection 的 `DataContainer`
- `Collect3D` 收集模型 forward 需要的字段

### 5.3 Radar 数据链路

radar 是当前项目的重要改造点：

```text
tools/data_converter/nuscenes_converter.py
  -> info["radars"]
  -> NuScenesDataset.get_data_info()
  -> data["radar"]
  -> LoadRadarPointsMultiSweeps
  -> RadarPoints
  -> DefaultFormatBundle3D
  -> model.forward(..., radar=...)
  -> BEVFusion.extract_features(radar, "radar")
```

如果运行 camera+radar 配置，需要使用带 radar 字段的 info pkl，例如：

```text
nuscenes_radar/nuscenes_radar_infos_train_radar.pkl
nuscenes_radar/nuscenes_radar_infos_val_radar.pkl
```

普通 nuScenes info pkl 可能没有 `radars` 字段，会导致 radar pipeline 或模型 forward 缺字段。

## 6. 模型结构

核心文件：

- `mmdet3d/models/fusion_models/bevfusion.py`
- `mmdet3d/models/builder.py`
- `mmdet3d/models/backbones/`
- `mmdet3d/models/necks/`
- `mmdet3d/models/vtransforms/`
- `mmdet3d/models/fusers/`
- `mmdet3d/models/heads/`

### 6.1 Registry 和 builder

`mmdet3d/models/builder.py` 定义了项目自有注册表：

| 注册表 | 用途 |
|---|---|
| `FUSIONMODELS` | 顶层融合模型，例如 `BEVFusion` |
| `VTRANSFORMS` | camera view transform，例如 `LSSTransform`、`AwareDBEVDepth` |
| `FUSERS` | 多模态融合模块，例如 `ConvFuser`、`AddFuser` |

同时复用 MMDetection 的：

- `BACKBONES`
- `NECKS`
- `HEADS`
- `LOSSES`

因此新增模块时通常要：

1. 在对应文件里加 `@XXX.register_module()`
2. 在对应目录的 `__init__.py` 中 import
3. 在 YAML 配置中把 `type` 指向新类名

### 6.2 BEVFusion 总体结构

`BEVFusion` 是顶层模型容器，负责组装和调度所有分支。

```text
camera images
  -> camera backbone
  -> camera neck
  -> vtransform
  -> camera BEV feature

LiDAR points
  -> voxelize / dynamic scatter
  -> lidar backbone
  -> lidar BEV feature

radar points
  -> voxelize / dynamic scatter
  -> radar backbone
  -> radar BEV feature

BEV features
  -> fuser
  -> decoder backbone
  -> decoder neck
  -> object/map head
```

训练时输出 loss 字典：

```text
loss/object/...
loss/map/...
loss/depth
stats/object/...
```

测试时输出：

```text
boxes_3d
scores_3d
labels_3d
masks_bev
```

### 6.3 Camera 分支

关键函数：

- `BEVFusion.extract_camera_features`

流程：

```text
img: [B, N, C, H, W]
  -> reshape to [B*N, C, H, W]
  -> camera backbone
  -> camera neck
  -> reshape back to [B, N, C, h, w]
  -> vtransform 投影到 BEV
```

常见 camera backbone：

- `ResNet`
- `VoVNet`
- `SwinTransformer` 相关配置可能来自 mmdet/mmcv 依赖

常见 view transform：

- `LSSTransform`
- `BEVDepth`
- `AwareBEVDepth`
- `DBEVDepth`
- `AwareDBEVDepth`

算法优化常见入口：

- 改 backbone：更换图像特征提取器
- 改 neck：优化多尺度融合
- 改 vtransform：深度估计、点云/雷达辅助深度、BEV pooling
- 改 depth loss：深度监督方式、mask、loss weight

### 6.4 LiDAR 分支

关键函数：

- `BEVFusion.extract_features(points, "lidar")`
- `BEVFusion.voxelize(points, sensor)`

流程：

```text
points
  -> Voxelization 或 DynamicScatter
  -> feats / coords / sizes
  -> lidar backbone
  -> BEV feature
```

`Voxelization` 和 `DynamicScatter` 区别：

| 方式 | 特点 |
|---|---|
| `Voxelization` | hard voxelization，每个 voxel 最多保留 `max_num_points` 个点 |
| `DynamicScatter` | 动态聚合，不固定每个 voxel 的点数量 |

算法优化常见入口：

- voxel size
- point cloud range
- max voxel 数
- LiDAR backbone，例如 `SparseEncoder`、`SECOND`
- 是否使用更多 sweeps
- 点云增强策略

### 6.5 Radar 分支

关键文件：

- `mmdet3d/models/backbones/radar_encoder.py`
- `configs/nuscenes/det/centerhead/lssfpn/camera+radar/default.yaml`

典型 radar encoder 配置：

```yaml
model:
  encoders:
    radar:
      voxelize_reduce: false
      voxelize:
        max_num_points: 20
        point_cloud_range: ${point_cloud_range}
        voxel_size: ${radar_voxel_size}
      backbone:
        type: RadarEncoder
        pts_voxel_encoder:
          type: RadarFeatureNet
          in_channels: 45
        pts_middle_encoder:
          type: PointPillarsScatter
          output_shape: [128, 128]
```

radar 分支基本是 PointPillars 风格：

```text
RadarPoints
  -> voxelize
  -> RadarFeatureNet
  -> PointPillarsScatter
  -> radar BEV feature
```

算法优化常见入口：

- `radar_use_dims`：选择哪些 radar 属性输入网络
- `radar_sweeps`：使用多少历史 radar 帧
- `radar_max_points`：最大 radar 点数
- `compensate_velocity`：是否补偿速度
- `filtering`：radar 点过滤策略
- `RadarFeatureNet`：点特征编码方式
- camera depth 中的 `use_points: radar`：用 radar 辅助图像深度

### 6.6 Fuser

关键文件：

- `mmdet3d/models/fusers/add.py`
- `mmdet3d/models/fusers/conv.py`

已有 fuser：

| 模块 | 逻辑 |
|---|---|
| `AddFuser` | 各模态先卷积到同一通道数，再相加，可训练时随机丢一路 |
| `ConvFuser` | 多模态 BEV 特征 channel concat，再接 Conv-BN-ReLU |

`ConvFuser` 伪代码：

```python
x = torch.cat(inputs, dim=1)
x = Conv2d(sum(in_channels), out_channels, 3, padding=1)
x = BN(x)
x = ReLU(x)
```

算法优化常见入口：

- attention/gating 融合
- radar-camera 深度引导融合
- 不同模态置信度估计
- 不同 BEV 尺度的融合
- fusion 前对齐误差补偿

### 6.7 Decoder 和 Heads

decoder 一般包括：

- BEV backbone，例如 `GeneralizedResNet`
- BEV neck，例如 `LSSFPN`、`SECONDFPN`

检测 head：

- `mmdet3d/models/heads/bbox/centerpoint.py`
- `mmdet3d/models/heads/bbox/transfusion.py`

分割 head：

- `mmdet3d/models/heads/segm/vanilla.py`

检测任务输出：

```text
heatmap / center / height / dim / rot / vel ...
  -> loss
  -> decode boxes
  -> NMS
```

分割任务输出：

```text
BEV feature
  -> per-class mask logits
  -> BCE/Focal loss
```

算法优化常见入口：

- 检测 head query 设计
- heatmap target 生成
- bbox coder
- NMS 策略
- 多任务 loss weight
- 分割 head 类别不平衡处理

## 7. View Transform 和 BEVDepth

关键文件：

- `mmdet3d/models/vtransforms/base.py`
- `mmdet3d/models/vtransforms/lss.py`
- `mmdet3d/models/vtransforms/depth_lss.py`
- `mmdet3d/models/vtransforms/aware_bevdepth.py`

LSS / BEVDepth 的大致逻辑：

```text
image feature
  -> predict depth distribution
  -> lift image feature to frustum
  -> use calibration transform frustum points to LiDAR/ego space
  -> splat/pool to BEV grid
```

`AwareDBEVDepth` 配置示例：

```yaml
model:
  encoders:
    camera:
      vtransform:
        type: AwareDBEVDepth
        bevdepth_downsample: 16
        depth_loss_factor: 3.0
        use_points: radar
        depth_input: one-hot
        height_expand: true
        add_depth_features: true
```

这表示：

- 使用 BEVDepth 风格深度估计
- 深度监督下采样倍率为 16
- 深度 loss 权重为 3.0
- 使用 radar 点辅助深度
- radar/lidar depth 输入编码为 one-hot
- 使用高度扩展和深度特征增强

`BEVFusion` 中通过 vtransform 类型判断是否启用深度 loss：

```python
self.use_depth_loss = vtransform_type in [
    "BEVDepth",
    "AwareBEVDepth",
    "DBEVDepth",
    "AwareDBEVDepth",
]
```

如果启用深度 loss，camera vtransform 需要返回：

```text
(bev_feature, depth_loss)
```

否则训练时会报 `Use depth loss is true, but depth loss not found`。

## 8. 配置到代码的对应关系

以 camera+radar 为例：

```yaml
model:
  type: BEVFusion
  encoders:
    lidar: null
    camera:
      backbone:
        type: ResNet
      neck:
        type: SECONDFPN
      vtransform:
        type: LSSTransform
    radar:
      voxelize:
        ...
      backbone:
        type: RadarEncoder
  fuser:
    type: ConvFuser
  decoder:
    backbone:
      type: GeneralizedResNet
    neck:
      type: LSSFPN
  heads:
    object:
      type: CenterHead 或 TransFusionHead
```

对应代码：

| 配置字段 | 构建函数 | 代码位置 |
|---|---|---|
| `model.type: BEVFusion` | `build_model` | `mmdet3d/models/fusion_models/bevfusion.py` |
| `encoders.camera.backbone` | `build_backbone` | `mmdet3d/models/backbones/` |
| `encoders.camera.neck` | `build_neck` | `mmdet3d/models/necks/` |
| `encoders.camera.vtransform` | `build_vtransform` | `mmdet3d/models/vtransforms/` |
| `encoders.radar.backbone` | `build_backbone` | `mmdet3d/models/backbones/radar_encoder.py` |
| `fuser` | `build_fuser` | `mmdet3d/models/fusers/` |
| `decoder.backbone` | `build_backbone` | `mmdet3d/models/backbones/` |
| `decoder.neck` | `build_neck` | `mmdet3d/models/necks/` |
| `heads.object` / `heads.map` | `build_head` | `mmdet3d/models/heads/` |

## 9. 做算法优化时优先看哪里

### 9.1 优化 camera-only 或 camera 主导模型

优先看：

- `configs/nuscenes/det/centerhead/lssfpn/camera/`
- `mmdet3d/models/vtransforms/`
- `mmdet3d/models/necks/`
- `mmdet3d/models/heads/bbox/centerpoint.py`

常见实验：

- 换 backbone
- 修改输入分辨率 `image_size`
- 修改 depth bins：`dbound`
- 修改 BEV 网格：`xbound/ybound/zbound`
- 加强 depth loss
- 修改 image augmentation

### 9.2 优化 camera+radar

优先看：

- `configs/nuscenes/det/centerhead/lssfpn/camera+radar/`
- `mmdet3d/datasets/pipelines/loading.py` 中的 `LoadRadarPointsMultiSweeps`
- `mmdet3d/models/backbones/radar_encoder.py`
- `mmdet3d/models/vtransforms/aware_bevdepth.py`
- `mmdet3d/models/fusers/`

常见实验：

- radar 属性维度选择
- radar sweeps 数量
- radar velocity compensation
- radar 点过滤
- radar 辅助深度估计
- attention/gated fusion
- radar BEV encoder 加深或轻量化

### 9.3 优化 camera+lidar

优先看：

- `configs/nuscenes/det/transfusion/secfpn/camera+lidar/`
- `mmdet3d/models/fusion_models/bevfusion.py`
- `mmdet3d/models/fusers/`
- `mmdet3d/models/backbones/sparse_encoder.py`
- `mmdet3d/models/heads/bbox/transfusion.py`

常见实验：

- LiDAR voxel size
- LiDAR sweeps 数量
- fuser 结构
- decoder 结构
- TransFusion head 参数
- NMS 策略

### 9.4 优化 BEV segmentation

优先看：

- `configs/nuscenes/seg/`
- `mmdet3d/datasets/pipelines/loading.py` 中的 `LoadBEVSegmentation`
- `mmdet3d/models/heads/segm/vanilla.py`

常见实验：

- BEV map resolution
- map 类别定义
- segmentation loss
- decoder 输出尺度
- camera/lidar/radar 多模态融合

## 10. 常见坑和注意事项

### 10.1 配置表达式会 eval

`mmdet3d/utils/config.py` 中：

```python
elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
    obj = eval(obj[2:-1], globals)
```

这让配置很灵活，但不要加载不可信 YAML。

### 10.2 radar 配置需要 radar info pkl

使用 camera+radar pipeline 时，`ann_file` 要指向带 radar 信息的 pkl。否则 dataset 中没有 `radar` 字段。

### 10.3 多模态特征 shape 必须对齐

fuser 默认假设所有输入 BEV feature 的空间尺寸一致：

```python
torch.cat(inputs, dim=1)
```

如果改了 `xbound/ybound/voxel_size/output_shape/downsample`，要检查 camera、lidar、radar 输出 BEV 尺寸是否一致。

### 10.4 `voxelize_reduce` 当前是共享属性

`BEVFusion.__init__` 中 lidar 和 radar 都写到 `self.voxelize_reduce`。如果同时开启 lidar 和 radar，且二者配置不同，后初始化的分支会覆盖前一个分支。这是潜在 bug。更稳妥的设计是改成：

```python
self.voxelize_reduce = {}
self.voxelize_reduce["lidar"] = ...
self.voxelize_reduce["radar"] = ...
```

然后在 `voxelize(points, sensor)` 中按 sensor 读取。

### 10.5 修改 pipeline 后要同步 Collect3D keys

新增模型 forward 参数时，需要同时检查：

- pipeline 是否产生该字段
- `DefaultFormatBundle3D` 是否格式化该字段
- `Collect3D.keys` 或 `meta_keys` 是否收集该字段
- `BEVFusion.forward` 是否接收该字段

### 10.6 深度 loss 分支有返回格式要求

如果 vtransform 类型在深度 loss 白名单中，forward 需要返回 feature 和 depth loss。否则训练时会报错。

## 11. 新增算法模块的最小步骤

### 11.1 新增 fuser

1. 在 `mmdet3d/models/fusers/` 新建文件，例如 `attention.py`
2. 使用 `@FUSERS.register_module()` 注册
3. 在 `mmdet3d/models/fusers/__init__.py` import
4. 在 YAML 中配置：

```yaml
model:
  fuser:
    type: YourAttentionFuser
    in_channels: [64, 64]
    out_channels: 64
```

### 11.2 新增 view transform

1. 在 `mmdet3d/models/vtransforms/` 新建或修改文件
2. 使用 `@VTRANSFORMS.register_module()` 注册
3. 在 `mmdet3d/models/vtransforms/__init__.py` import
4. 确保 forward 入参和 `BEVFusion.extract_camera_features` 调用兼容
5. 如果使用深度 loss，确保返回 `(bev_feature, depth_loss)`

### 11.3 新增 backbone

1. 在 `mmdet3d/models/backbones/` 新增模块
2. 使用 MMDetection 的 `@BACKBONES.register_module()` 注册
3. 在 `mmdet3d/models/backbones/__init__.py` import
4. 在 YAML 中把 `type` 指向新类

### 11.4 新增数据字段

1. 在 dataset 或 pipeline 中产生字段
2. 在 `DefaultFormatBundle3D` 中转 tensor 或 DataContainer
3. 在 `Collect3D` 中收集
4. 在模型 forward 中新增参数
5. 修改相关配置的 `keys` / `meta_keys`

## 12. 调试建议

### 12.1 先打印配置

`tools/test.py` 会 `print(cfg)`。训练时也会 dump 到 run dir：

```text
cfg.run_dir/configs.yaml
```

遇到配置不生效时，先看最终 dump 后的配置，而不是只看目标 YAML。

### 12.2 先确认 batch 字段

改 pipeline 后，优先确认 dataloader 输出字段是否符合模型 forward。重点字段：

```text
img
points
radar
camera2ego
lidar2ego
lidar2camera
lidar2image
camera_intrinsics
camera2lidar
img_aug_matrix
lidar_aug_matrix
metas
depths
gt_bboxes_3d
gt_labels_3d
gt_masks_bev
```

### 12.3 先确认 BEV feature shape

在改 view transform、radar encoder、fuser 时，最常见的问题是 shape 不一致。建议临时打印：

```python
print(sensor, feature.shape)
```

位置：

```text
mmdet3d/models/fusion_models/bevfusion.py
  -> forward_single
  -> features.append(feature)
```

### 12.4 先做小规模 sanity check

在正式训练前建议：

- 单卡
- `samples_per_gpu: 1`
- 少量数据
- 先跑几 iter 观察 loss 是否为 nan
- 再开多卡和完整训练

## 13. 一张总流程图

```text
YAML config
   |
   v
tools/train.py 或 tools/test.py
   |
   v
build_dataset ----------------------+
   |                                |
   v                                |
NuScenesDataset                     |
   |                                |
   v                                |
pipeline: load/augment/format       |
   |                                |
   +---------- batch ---------------+
                                    |
                                    v
                              build_model
                                    |
                                    v
                              BEVFusion
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
   camera encoder             lidar encoder              radar encoder
          |                         |                         |
          v                         v                         v
   camera BEV feature         lidar BEV feature          radar BEV feature
          |                         |                         |
          +-------------------------+-------------------------+
                                    |
                                    v
                                  fuser
                                    |
                                    v
                                 decoder
                                    |
                                    v
                         object head / map head
                                    |
                                    v
                            loss 或 eval result
```

## 14. 最短上手路径

如果目标是快速开始做算法改进，可以按这个顺序：

1. 选定一个实验配置，例如 camera+radar 的 `resnet50/default.yaml` 或 `dlss.yaml`
2. 用最终 dump 的 `configs.yaml` 确认完整模型和 pipeline
3. 读 `BEVFusion.forward_single`，弄清楚该配置实际启用了哪些 encoder
4. 读对应 encoder：
   - camera: `vtransforms/`
   - lidar: `backbones/sparse_encoder.py` 或相关配置
   - radar: `backbones/radar_encoder.py`
5. 读 fuser：`fusers/add.py` 或 `fusers/conv.py`
6. 读 head：`heads/bbox/centerpoint.py`、`heads/bbox/transfusion.py` 或 `heads/segm/vanilla.py`
7. 从配置可控参数开始实验，再进入模块代码改结构

