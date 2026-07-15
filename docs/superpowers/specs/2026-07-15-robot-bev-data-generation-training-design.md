# 统一 Robot BEV 数据生成与训练设计

状态：已确认设计

日期：2026-07-15

## 1. 目标

为本仓库建立一套与具体数据源解耦的 Robot BEV 数据标准、虚拟数据生成工具和 BEVFusion 训练链路。当前第一个数据源是通过 Habitat-Sim 0.2.2 渲染的 Replica v1 PTex 场景，后续可增加 HM3D、Gibson 或其他 Habitat-Sim 数据源，也可接入真实机器人数据。

所有数据源必须统一以下内容：

- 目录结构与相对路径规则；
- train/val/test 场景级划分；
- 传感器坐标系、单位和标定矩阵语义；
- 点云字段和历史帧时间定义；
- BEV 网格范围、分辨率和类别顺序；
- 观测范围与有效监督范围；
- 结构、数值、几何和训练前验证规则；
- 转换到 BEVFusion 的唯一实现。

正确训练优先于兼容旧输出。现有 90 帧测试数据和远端 18×600 帧数据不提供迁移工具，统一删除后用新版生成器重新生成。

## 2. 总体架构

采用“数据源适配器 + 标准数据集 + 框架转换器”的分层架构：

```text
Replica / HM3D / Gibson / 真实机器人
                  │
                  ▼
          数据源适配器或生成器
                  │
                  ▼
       robot_bev_dataset 标准数据
                  │
          ┌───────┴────────┐
          ▼                ▼
   强制数据与几何校验   人工诊断图检查
          │                │
          └───────┬────────┘
                  ▼
          唯一 BEVFusion 转换器
                  │
                  ▼
      各数据根目录独立的框架索引
                  │
                  ▼
       单数据集或 ConcatDataset 训练
```

源生成器不得生成 MMDetection3D 或 NuScenes 专属字段。`cams`、四元数、空 3D 检测标注和历史 sweep 变换等字段只由 BEVFusion 转换器生成。

不同来源的数据保存在独立根目录中，不为联合训练复制或物理合并原始文件，也不创建跨根目录的合并 pickle。联合训练在配置层使用 `ConcatDataset`。

## 3. 数据生成工具目录

数据生成代码从被 Git 忽略的 `data/generate_mydata/` 移到可追踪的顶层目录：

```text
data_generation/
├── __init__.py
└── robot_bev/
    ├── README.md
    ├── __init__.py
    ├── schema.py
    ├── writer.py
    ├── validator.py
    ├── geometry_checks.py
    │
    ├── sources/
    │   ├── __init__.py
    │   ├── habitat_common.py
    │   └── replica.py
    │
    ├── cli/
    │   ├── __init__.py
    │   ├── generate_replica.py
    │   └── validate_dataset.py
    │
    ├── configs/
    │   ├── replica_scenes.txt
    │   └── replica_splits.example.json
    │
    └── docs/
        ├── schema_v3.md
        ├── habitat_replica.md
        ├── add_new_source.md
        └── quality_checks.md
```

模块边界如下：

- `schema.py`：定义 schema 常量、数据结构和不变量，不依赖 Habitat-Sim、MMCV、MMDetection3D 或 BEVFusion。
- `writer.py`：原子写入帧文件、manifest、场景索引、根索引和元数据，不包含数据源特有逻辑。
- `validator.py`：执行结构与数值验证，可在无 Habitat-Sim 的训练环境中运行。
- `geometry_checks.py`：生成相机投影、BEV 朝向和多帧点云对齐诊断产物。
- `sources/habitat_common.py`：封装 Habitat-Sim 传感器、轨迹、位姿、深度反投影和公共坐标变换。
- `sources/replica.py`：仅实现 Replica v1 PTex 文件检查、NavMesh、场景加载和语义类别映射。
- `cli/`：只负责参数解析和编排，不重复 schema、writer 或 source 逻辑。

未来仍通过 Habitat-Sim 渲染的新数据集只需新增 source adapter。非 Habitat 数据源也可以复用同一 schema、writer 和 validator。

## 4. 标准数据集根目录

每个来源使用独立根目录：

```text
data/replica_robot_bev_v3/
data/hm3d_robot_bev_v3/
data/gibson_robot_bev_v3/
```

根目录结构固定为：

```text
<data_root>/
├── dataset_metadata.json
├── splits.json
├── robot_infos_train.pkl
├── robot_infos_val.pkl
├── robot_infos_test.pkl
├── bevfusion_infos_train.pkl
├── bevfusion_infos_val.pkl
├── bevfusion_infos_test.pkl
├── multi_scene_summary.json
│
├── <scene_id>/
│   ├── images/
│   ├── points/
│   ├── bev_masks/
│   ├── bev_observed_masks/
│   ├── calib/
│   │   ├── camera_intrinsic.txt
│   │   ├── camera2base.txt
│   │   └── lidar2base.txt
│   ├── poses/
│   │   └── poses.txt
│   ├── manifest.jsonl
│   ├── metadata.json
│   ├── summary.json
│   ├── scene_infos.pkl
│   ├── bev_supervision_masks/  # 可选
│   ├── depths/                 # 可选
│   ├── semantics/              # 可选
│   └── visualizations/         # 可选
│
└── <another_scene_id>/
```

`robot_infos_<split>.pkl` 是与训练框架无关的标准根索引。`bevfusion_infos_<split>.pkl` 是可随时删除和重建的框架派生产物。`manifest.jsonl` 是可人工检查并可用于重建索引的逐帧来源记录。

所有 manifest 和 pickle 中的文件路径必须是相对于 `<data_root>` 的 POSIX 路径。禁止绝对路径，禁止依赖训练进程当前工作目录解析路径。

## 5. Schema 版本与根元数据

根元数据至少包含：

```json
{
  "schema_name": "robot_bev_dataset",
  "schema_version": 3,
  "dataset_id": "replica_robot_bev_v3",
  "source_type": "simulation",
  "source_dataset": "replica_v1",
  "generator": {
    "name": "habitat_replica_robot_bev",
    "version": "3"
  },
  "map_classes": [
    "floor",
    "carpet",
    "obstacle",
    "wall",
    "furniture",
    "other"
  ],
  "bev": {
    "xbound": [0.0, 3.0, 0.02],
    "ybound": [-1.5, 1.5, 0.02],
    "shape": [6, 150, 150],
    "encoding": "uint8_multihot",
    "observed_mask_shape": [150, 150]
  },
  "points": {
    "dtype": "float32",
    "dimensions": ["x", "y", "z", "intensity", "time"]
  }
}
```

`dataset_id` 在一次联合训练所引用的数据根目录之间必须唯一。允许的 `source_type` 至少包括 `simulation`、`real_robot` 和 `converted_dataset`。

以下变化必须增加 schema 版本：

- 坐标系或矩阵语义变化；
- 必需字段变化；
- 类别含义或顺序变化；
- BEV shape、范围或分辨率变化；
- 点云维度、dtype 或单位变化；
- 路径解释规则变化。

仅增加向后兼容的可选字段不需要增加 schema 版本。

## 6. 场景划分

`splits.json` 只保存场景 ID：

```json
{
  "train": ["apartment_0", "office_0"],
  "val": ["office_1"],
  "test": ["office_4"]
}
```

规则如下：

- 一个场景只能出现在一个 split；
- 同一轨迹的相邻帧不得拆到不同 split；
- 每个场景 ID 必须对应且只对应一个场景目录；
- split 中不得出现重复或不存在的场景；
- test 只用于最终评估，不参与超参数选择。

当前 18 场景、每场景 600 帧的建议划分为 14/2/2 个场景，对应 train 8400 帧、val 1200 帧、test 1200 帧。

## 7. 标准帧记录

每个 `robot_infos_<split>.pkl` 使用以下外层结构：

```python
{
    "metadata": {...},
    "infos": [frame_record, ...],
}
```

标准帧记录包含：

```python
{
    "dataset_id": "replica_robot_bev_v3",
    "scene_id": "office_0",
    "frame_id": 12,
    "token": "replica_robot_bev_v3:office_0:000012",
    "prev_token": "replica_robot_bev_v3:office_0:000011",
    "timestamp": 2200000,

    "image_path": "office_0/images/000012.png",
    "lidar_path": "office_0/points/000012.bin",
    "bev_mask_path": "office_0/bev_masks/000012.npy",
    "bev_observed_mask_path": "office_0/bev_observed_masks/000012.npy",
    "bev_supervision_mask_path": null,

    "class_validity": np.ones((6,), dtype=np.uint8),
    "cam_intrinsic": np.ndarray((3, 3), dtype=np.float32),
    "camera2base": np.ndarray((4, 4), dtype=np.float32),
    "lidar2base": np.ndarray((4, 4), dtype=np.float32),
    "T_map_base": np.ndarray((4, 4), dtype=np.float32),
    "pose_valid": True,

    "depth_path": "office_0/depths/000012.png",
    "semantic_path": "office_0/semantics/000012.png"
}
```

`depth_path`、`semantic_path` 和 `bev_supervision_mask_path` 是可选字段，其他列出的字段为必需字段。

JSON manifest 中的 numpy 数组按普通 JSON list 写出；pickle 索引恢复为以上明确的 numpy dtype 和 shape。可选路径可以省略或写为 null，但同一数据根必须使用一种一致形式。

附加规则：

- `token` 在一个数据根目录内唯一；
- `timestamp` 是整数微秒，并在场景内严格递增；
- 第一帧 `prev_token` 为空，其他帧只指向同场景上一帧；
- `pose_valid` 必须存在，正式训练数据必须为 true；
- 原始记录不保存 BEVFusion sweeps，转换器从位姿和标定统一计算；
- 所有矩阵使用 `float32` 齐次变换。

## 8. 坐标、标定和传感器契约

所有变换使用列向量。名称 `A2B` 表示 `T_B_from_A`：

```text
p_base = camera2base @ p_camera
p_base = lidar2base  @ p_lidar
p_map  = T_map_base @ p_base
```

机器人 base 和 LiDAR 坐标为右手系：

```text
x: forward
y: left
z: up
```

相机使用 OpenCV optical 坐标：

```text
x: right
y: down
z: forward
```

`camera2base` 已经表示 `T_base_from_camera_optical`。任何消费者均不得再次应用 OpenGL 到 OpenCV 的轴变换。

RGB 必须在导出前完成去畸变，`cam_intrinsic` 必须与保存图像的分辨率一致。坐标、平移和深度统一使用米。

点云文件是连续排列的 `float32 [x, y, z, intensity, time]`：

- 当前帧 `time=0`；
- 历史点的 time 为“当前时间减历史时间”的秒数；
- intensity 推荐归一化到 `[0,1]`；
- 无强度来源写 0，不得改变点维度。

历史 LiDAR 到当前 LiDAR 的变换固定为：

```text
T_cur_lidar_from_hist_lidar
  = inverse(T_map_base_cur @ lidar2base_cur)
  @ (T_map_base_hist @ lidar2base_hist)
```

## 9. BEV 网格和六类语义

BEV 网格固定为：

```text
xbound = [0.0, 3.0, 0.02]
ybound = [-1.5, 1.5, 0.02]
row = floor((x - 0.0) / 0.02)
col = floor((y + 1.5) / 0.02)
```

语义 mask 固定为 `[6,150,150]`、`uint8`、二值 multi-hot，通道顺序固定为：

1. `floor`：可通行硬质地面；
2. `carpet`：地毯、地垫或 rug，可与 floor 重叠；
3. `obstacle`：阻碍通行的几何体；
4. `wall`：墙体，可与 obstacle 重叠；
5. `furniture`：桌、椅、床、柜、沙发等，可与 obstacle 重叠；
6. `other`：已识别但不属于以上语义组的物体，可与 obstacle 重叠。

`unknown` 不作为训练类别。它只在可视化或下游应用中按 `1 - observed_mask` 派生。

未来数据源必须提供显式的源类别到六类标准类别映射。不得将“源数据没有标注某类”解释为该类负样本。

## 10. 观测 mask 与监督 mask

为避免给每一帧重复保存一个通常等于 observed mask 广播结果的 `[6,150,150]` 文件，采用组合式监督定义。

必需的 `bev_observed_mask`：

- shape `[150,150]`；
- dtype `uint8`；
- 值为 `{0,1}`；
- 1 表示该网格被当前传感器可靠观测；
- 0 表示未观测或观测不可靠。

必需的 `class_validity`：

- shape `[6]`；
- dtype `uint8`；
- 值为 `{0,1}`；
- 1 表示该帧中该类别可被当前数据源可靠监督；
- 0 表示该类别不可被可靠标注，既不作为正样本也不作为负样本。

可选的 `bev_supervision_mask`：

- shape `[6,150,150]`；
- dtype `uint8`；
- 值为 `{0,1}`；
- 仅当同一类别在不同空间区域具有不同标注可靠性时保存。

最终训练和评估使用的有效 mask 为：

```python
effective = observed_mask[None, :, :] * class_validity[:, None, None]
if per_class_supervision_mask is not None:
    effective = effective * per_class_supervision_mask
```

当前 Replica 数据六类均可监督，因此写入 `class_validity=[1,1,1,1,1,1]`，不生成逐类 supervision mask。训练时仅广播 observed mask，不产生重复存储。

所有六个语义通道在 observed mask 外必须为 0。

## 11. Replica/Habitat-Sim 生成器

现有 `data/generate_mydata/robot_bev_closed_loop.py` 中已经验证的以下逻辑保留并拆分：

- Habitat-Sim 0.2.2 与 Replica v1 PTex 前置检查；
- RGB、depth 和 semantic sensor 配置；
- NavMesh 初始化和轨迹控制；
- Habitat 状态到 base 位姿转换；
- 深度反投影为 base/LiDAR 点云；
- 前视射线 observed mask；
- floor、carpet、obstacle、wall、furniture、other 标签生成；
- 原子帧写入和断点续跑。

Replica 特有逻辑放入 `sources/replica.py`，Habitat 通用逻辑放入 `sources/habitat_common.py`。帧产物和索引统一由 `writer.py` 写入。

旧生成器的 `--num-sweeps` 参数从生成阶段移除。标准原始数据只保存逐帧 pose 和 calibration，历史帧数量由 BEVFusion 转换器或训练配置统一决定。

正式生成要求 Habitat-Sim 精确版本为 0.2.2。版本不匹配只允许执行非生产诊断，不得生成正式训练数据。

## 12. Writer、断点续跑和失败策略

每一帧必须在所有必需文件原子写入完成后才追加 `manifest.jsonl`。中断后只能从 manifest 中最后一个完整帧继续。

每个场景元数据保存生成参数 fingerprint。以下参数变化后禁止在原目录续跑：

- schema 版本；
- 数据源或场景版本；
- 图像尺寸、FOV 或相机外参；
- BEV 范围和分辨率；
- 类别顺序或类别映射；
- 深度、点云或障碍物阈值；
- split；
- observed/supervision mask 定义。

正式多场景生成中，单场景失败必须记录到根 summary；命令最终以失败状态退出。失败的数据根目录不得被转换器或训练配置接受。

不在旧目录中边生成边覆盖。旧输出确认废弃后删除，再使用全新的输出目录生成标准数据。

## 13. 强制数据验证

验证是转换和训练的硬门槛。

### 13.1 结构验证

- 根元数据 schema 名称和版本受支持；
- split 场景集合互斥且完整；
- 场景 ID、目录和索引一致；
- token 唯一，时间戳在场景内递增；
- prev_token 不跨场景；
- 每个必需路径存在且位于数据根目录内；
- RGB、点云、标签、observed mask 和 pose 帧数一致；
- metadata 类别顺序与固定六类完全一致。

### 13.2 数值验证

- 语义 mask 为 `[6,150,150]`、`uint8`、二值；
- observed mask 为 `[150,150]`、`uint8`、二值；
- class_validity 为 `[6]`、`uint8`、二值；
- 可选 supervision mask 为 `[6,150,150]`、`uint8`、二值；
- observed mask 外所有语义标签为 0；
- 点云字节数可被 `5 * sizeof(float32)` 整除；
- 点、内参和变换矩阵值均有限；
- 齐次矩阵最后一行合法；
- 旋转矩阵近似正交且行列式接近 +1；
- 相机内参适用于保存的图像尺寸；
- pose_valid 为 true。

### 13.3 几何验证

- 将 LiDAR 点投影到 RGB，并生成 overlay；
- 将 LiDAR 点投影到 BEV，验证前/左方向；
- 将历史 sweeps 变换到当前帧，检查静态结构对齐；
- 将相机 frustum 投影到 BEV，检查标签覆盖方向；
- 每种新数据源至少人工确认一个场景的诊断产物。

### 13.4 质量警告

以下问题生成警告和统计报告，但不自动修改数据：

- 某类在一个 split 中极度稀疏或完全缺失；
- observed 覆盖率异常低；
- 点数明显偏离该数据集分布；
- intensity 不在推荐范围；
- 场景轨迹覆盖不足或碰撞比例异常。

校验错误必须包含 dataset ID、scene ID、frame ID、字段名、期望值和实际值。禁止静默修正坐标系、矩阵、类别或路径。

## 14. BEVFusion 转换器

唯一转换器放在：

```text
tools/data_converter/robot_bev_converter.py
```

职责包括：

1. 调用 validator 验证一个标准数据根和 split；
2. 通过显式 dataset root 解析相对路径；
3. 直接使用 OpenCV `camera2base`，不得再次翻转相机轴；
4. 从当前/历史 pose 和 lidar2base 计算 sweeps；
5. 生成 NuScenesDataset 当前实现需要的 camera、ego、LiDAR 和四元数字段；
6. 保留语义、observed、class validity 和可选 supervision mask 路径；
7. 生成空 3D 检测标注；
8. 保持 token 为 `<dataset_id>:<scene_id>:<frame_id>`；
9. 在派生 metadata 中记录转换器版本和源 schema；
10. 确定性写出 `bevfusion_infos_<split>.pkl`。

转换器不得修改标签、猜测坐标系、替换非法矩阵或默认跳过坏帧。

## 15. BEVFusion 数据加载

在保留现有 NuScenesDataset 主体的前提下增加标准 robot BEV 支持：

- 显式选项控制相对路径按 `dataset_root` 解析，默认关闭以保持 nuScenes 行为；
- 从 info 中传递 observed mask、class validity 和可选 supervision mask 路径；
- pipeline 同时加载 `gt_masks_bev` 和组合后的 `gt_supervision_mask_bev`；
- formatting 和 `Collect3D` 将两个 tensor 以 batch 维堆叠；
- fusion model forward 接收监督 mask；
- 推理评估输出保留 GT mask 和监督 mask。

`depths` 在 fusion model forward 中改为可选参数。当前 `LSSTransform` 配置不生成无用的 `GTDepth`；只有明确启用 BEVDepth 类深度监督的配置才在 pipeline 中添加深度 target。

训练输入为：

```text
前视 RGB
当前 LiDAR 点云
最多 5 个历史 sweeps
六通道 BEV 标签
六通道有效监督 mask
```

仅做分割时不使用检测类别均衡的 `CBGSDataset`，直接使用基础 dataset。空 3D 检测标注仅用于兼容现有数据结构，不进入 loss。

## 16. Masked loss 和评估

分割 head 必须先计算不做 reduction 的逐像素 focal loss：

```python
raw_loss = focal_loss(logits, target, reduction="none")
valid = supervision_mask.float()
loss = (raw_loss * valid).sum() / valid.sum().clamp_min(1.0)
```

每个类别单独报告 loss。某一 batch 中某类有效监督像素为 0 时，该类 loss 返回与计算图连接的 0，不产生 NaN。

IoU 的 TP、FP 和 FN 只在 supervision mask 为 1 的位置累计。评估输出至少包含：

- 每类 `IoU@0.50`；
- `mIoU@0.50`；
- 用于诊断的多个阈值 IoU；
- 每类有效监督像素数量。

模型选择使用固定阈值 `mIoU@0.50`，不得使用每类阈值扫描后的最大 IoU 作为主模型选择指标。test split 只在最终模型确定后评估一次。

## 17. Robot BEV 训练配置

新增独立配置层级：

```text
configs/robot_bev/
├── default.yaml
└── seg/
    └── camera_lidar_lss.yaml
```

初始配置固定：

- camera + LiDAR 融合；
- 前视单相机；
- 最多 5 个历史 sweeps；
- label 输出范围 `[0,3.0] × [-1.5,1.5]`，分辨率 0.02 m；
- 六类 sigmoid/focal 分割；
- 不使用 `CBGSDataset`；
- 关闭 3D 旋转、缩放和平移；
- 关闭 3D flip；
- 图像增强仅使用会同步更新相机增广矩阵的变换；
- 第一阶段按样本数自然采样，不加数据集权重；
- AdamW 和梯度裁剪沿用现有可靠设置，具体 batch size 由远端显存决定。

在 BEV 标签和 supervision mask 能同步变换之前，不启用 BEV 相关 3D augmentation。

## 18. 预训练权重加载

正式训练采用 nuScenes `bevfusion-seg.pth` 微调，但不能直接使用现有无过滤的 `--load_from`。

选择性加载必须同时检查参数名称和 shape：

- 尽量复用 camera backbone、camera neck、兼容的 view-transform 权重、LiDAR encoder 和 decoder；
- 几何网格参数 `dx`、`bx`、`nx`、`frustum` 始终使用当前配置重新生成；
- shape 不匹配的 fuser 参数重新初始化；
- 分割 head 中间特征层可在 shape 匹配时复用；
- 最后一层六通道分类卷积无条件跳过并重新初始化，因为 nuScenes 六通道语义与 robot BEV 六类不同；
- 所有跳过、缺失和成功加载的参数按模块汇总输出；
- 配置声明必须加载的核心模块如果完全没有匹配权重，立即报错。

微调时全模型参与训练。预训练 backbone 使用较小学习率，新初始化的 fuser 和分割输出层使用正常学习率。

## 19. 多数据集联合训练

每个数据根目录独立执行 validate 和 convert，然后通过 `ConcatDataset` 联合：

```yaml
data:
  train:
    type: ConcatDataset
    separate_eval: true
    datasets:
      - type: NuScenesDataset
        dataset_root: data/replica_robot_bev_v3/
        ann_file: data/replica_robot_bev_v3/bevfusion_infos_train.pkl
        resolve_relative_paths: true
      - type: NuScenesDataset
        dataset_root: data/hm3d_robot_bev_v3/
        ann_file: data/hm3d_robot_bev_v3/bevfusion_infos_train.pkl
        resolve_relative_paths: true
```

联合前必须验证所有根目录具有相同的 schema 版本、类别顺序、BEV 网格、坐标契约、点云维度和单位。

初始训练按样本数自然采样。只有实际统计证明来源不平衡影响训练后，才引入 dataset weights。

## 20. 训练前门槛和正式流程

统一流程为：

```text
generate
  -> validate
  -> convert
  -> inspect geometry
  -> single-sample pipeline
  -> one-batch forward/backward
  -> 16-32 frame overfit
  -> full train
  -> validation model selection
  -> one final test
```

训练前必须通过：

1. 完整结构和数值验证；
2. 至少一个场景的相机投影、BEV 朝向和 sweeps 人工检查；
3. 单样本 pipeline 可完整加载；
4. 单 batch 前向/反向的 loss 和梯度均有限；
5. 16～32 帧小样本训练能够明显降低 loss，并产生方向正确的预测。

90 帧规模只用于打通链路和小样本过拟合。正式远端训练使用重新生成的 8400 train / 1200 val / 1200 test 帧。

## 21. 测试策略

### 21.1 单元测试

- schema 常量和版本；
- 相对路径规范化与越界拒绝；
- split 互斥；
- mask shape、dtype 和二值检查；
- effective supervision mask 组合；
- 变换矩阵合法性；
- history-to-current sweep 公式；
- 选择性 checkpoint 参数过滤；
- masked focal loss 的归一化和零有效像素行为；
- masked IoU 只统计有效区域。

### 21.2 集成测试

- 使用少量合成文件写出标准数据集；
- validator 接受合法数据并拒绝逐项损坏的数据；
- converter 确定性生成 BEVFusion 索引；
- dataset pipeline 返回正确 tensor shape；
- 一个 batch 完成前向和反向；
- 推理结果携带评估所需 GT 和监督 mask。

### 21.3 几何测试

- camera optical 点投影到正确图像象限；
- base/LiDAR 的 x-forward、y-left 与 BEV row/col 一致；
- 当前和历史静态点在 sweep 对齐后重合；
- 诊断图与生成器可视化方向一致。

### 21.4 训练测试

- 所有 loss 有限；
- 有效区域外 target 改变不影响 loss 和 IoU；
- 小样本 overfit loss 明显下降；
- 保存和恢复 checkpoint 后指标一致。

## 22. 文档要求

数据生成工具必须提供：

- Replica/Habitat-Sim 0.2.2 环境与 PTex 前置条件；
- 单场景 smoke 和多场景正式生成命令；
- 断点续跑限制；
- schema 与坐标系说明；
- 如何新增 Habitat-Sim 数据源适配器；
- 数据校验和几何诊断说明；
- 从生成数据到 BEVFusion 训练的完整命令链；
- 远端 18×600 帧生成和训练操作说明。

## 23. 非目标

本轮不包含：

- 新增或修改六类语义；
- 真实 ToF/LiDAR 噪声和 domain randomization；
- BEV 标签同步 3D augmentation；
- 多数据集采样权重优化；
- 新模型结构或新融合算法；
- 自动执行远端 10800 帧生成；
- 自动启动完整远端训练；
- 旧数据迁移和向后兼容工具。

## 24. 验收条件

设计实现完成需要同时满足：

1. 新工具目录被 Git 跟踪，旧 `data/generate_mydata` 不再是正式代码来源；
2. Replica 生成器通过公共 schema、writer 和 validator 产出标准数据；
3. 生成器支持安全断点续跑并拒绝参数不一致的目录；
4. 数据根只使用可迁移的相对路径；
5. 六类、BEV 网格、坐标系和 split 契约被自动验证；
6. observed/class-validity/可选 per-class mask 正确组合且不重复存储常见 mask；
7. 相机、LiDAR、BEV 和历史 sweeps 几何诊断一致；
8. BEVFusion 转换器确定性生成三个 split 的索引；
9. invalid 区域不贡献 loss 或 IoU；
10. nuScenes 预训练权重按名称和 shape 选择性加载，六类输出层重新初始化；
11. 单 batch 前后向有限，16～32 帧可以被明显过拟合；
12. 重新生成的 18 场景数据可按 8400/1200/1200 流程训练、验证和最终测试；
13. 新增另一个 Habitat-Sim source adapter 时无需复制 schema、writer 或 BEVFusion 转换逻辑。
