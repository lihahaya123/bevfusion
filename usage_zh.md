# Robot BEV 数据生成与转换使用说明

本文档说明如何从 Habitat-Sim/Replica 渲染标准 `robot_bev_dataset v3`
数据，转换成 BEVFusion 可读取的训练索引，并使用当前 RobotBEV 训练配置
打通训练链路。

整体流程如下：

```text
Habitat-Sim/Replica 渲染
  -> robot_bev_dataset v3 标准数据
  -> 严格校验与几何诊断
  -> BEVFusion infos 转换
  -> RobotBEVDataset 读取
  -> BEVFusion masked BEV segmentation 训练
```

## 代码位置

数据生成工具位于：

```text
data_generation/robot_bev/
```

常用入口：

```text
data_generation/robot_bev/cli/generate_replica.py
data_generation/robot_bev/cli/validate_dataset.py
tools/data_converter/robot_bev_converter.py
```

关键模块：

```text
data_generation/robot_bev/schema.py
data_generation/robot_bev/writer.py
data_generation/robot_bev/validator.py
data_generation/robot_bev/geometry_checks.py
data_generation/robot_bev/sources/habitat_common.py
data_generation/robot_bev/sources/replica.py
```

## 环境准备

数据渲染使用 `habitat022` 环境：

```bash
conda activate habitat022
cd /home/lihahaya/workspace/hikvision/bevfusion
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

准备 Replica 数据集配置路径：

```bash
export REPLICA_CONFIG=/mnt/u/ubuntu/workspace/dataset/HIKVISION/replica/replica.scene_dataset_config.json
export OUTPUT_ROOT=/home/lihahaya/workspace/hikvision/bevfusion/data/replica_robot_bev_v3_repair
```

`REPLICA_CONFIG` 必须指向原始 Replica v1 PTex 的
`replica.scene_dataset_config.json`。生成器会检查 render mesh、PTex 贴图、
semantic mesh、`info_semantic.json`、navmesh 和 stage 配置。

## 生成 90 帧链路测试数据

这一步用于打通训练输入链路。它会渲染 9 个场景，每个场景 10 帧：
train 70 帧、val 10 帧、test 10 帧。

```bash
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset "$REPLICA_CONFIG" \
  --dataset-id replica_robot_bev_v3 \
  --scenes hotel_0 office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 \
  --split-file data_generation/robot_bev/configs/replica_splits.example.json \
  --output-dir "$OUTPUT_ROOT" \
  --num-frames 10 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh
```

当前本地 90 帧测试数据示例位置：

```text
data/replica_robot_bev_v3/
```

## 校验数据

先校验完整根目录，再分别校验 train、val、test。完整根目录校验用于确认
metadata、splits、scene summary 和 root index 一致；单独 split 校验用于确认
对应训练索引和样本内容可用。

```bash
python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT"

python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split train

python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split val

python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split test
```

校验成功时 JSON 输出中应包含：

```text
"valid": true
```

90 帧小数据中，val/test 可能出现某些类别计数为 0 的 warning。这类 warning
不等于格式错误，但训练前需要人工确认是否符合当前小样本的预期。

## 生成几何诊断图

建议每个 split 至少抽一个场景和帧做几何诊断：

```bash
python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split train \
  --geometry-scene office_0 \
  --geometry-frame 5

python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split val \
  --geometry-scene office_1 \
  --geometry-frame 5

python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split test \
  --geometry-scene office_4 \
  --geometry-frame 5
```

诊断图会写入：

```text
$OUTPUT_ROOT/diagnostics/<scene_id>/
```

重点看三类图：

```text
*_overview.png
*_rgb_point_overlay.png
*_bev_overlay.png
*_aligned_sweeps.png
```

`*_overview.png` 是单帧总览图，会拼接 RGB、深度伪彩、语义 ID 伪彩、
BEV label、observed mask 和 BEV+点云叠加图。其余三张分别用于检查相机
投影、BEV 方向和历史帧对齐。文件存在不代表几何正确，需要人工确认
x-forward/y-left、相机投影和 sweep 对齐是否符合预期。

本地可直接打开 PNG：

```bash
xdg-open "$OUTPUT_ROOT/diagnostics/office_0/000000_overview.png"
```

无图形界面的远端服务器可以把 diagnostics 目录拷回本地查看，或者用 VS Code /
Cursor 的远程文件预览打开 PNG。

## 转换为 BEVFusion 训练索引

校验通过后，执行转换器：

```bash
python tools/data_converter/robot_bev_converter.py \
  --root "$OUTPUT_ROOT" \
  --split all \
  --max-sweeps 5
```

输出文件位于数据根目录：

```text
$OUTPUT_ROOT/bevfusion_infos_train.pkl
$OUTPUT_ROOT/bevfusion_infos_val.pkl
$OUTPUT_ROOT/bevfusion_infos_test.pkl
```

转换器不会复制图片、点云或 BEV mask，只会基于标准
`robot_infos_<split>.pkl` 生成 BEVFusion 风格索引。训练时仍然以
`$OUTPUT_ROOT` 作为数据根目录读取相对路径。

## 训练前数据目录应包含

转换完成后，训练输入数据根目录至少应包含：

```text
<root>/
  dataset_metadata.json
  splits.json
  multi_scene_summary.json
  robot_infos_train.pkl
  robot_infos_val.pkl
  robot_infos_test.pkl
  bevfusion_infos_train.pkl
  bevfusion_infos_val.pkl
  bevfusion_infos_test.pkl
  <scene_id>/
    images/
    points/
    bev_masks/
    bev_observed_masks/
    calib/
    poses/
    manifest.jsonl
    scene_infos.pkl
    robot_infos_<split>.pkl
```

其中训练监督相关字段包括：

```text
bev_mask_path
bev_observed_mask_path
class_validity
bev_supervision_mask_path  # 可选
```

有效训练 mask 规则是：

```text
observed_mask[None, :, :]
  * class_validity[:, None, None]
  * optional_per_class_supervision_mask
```

不要把未观测区域或 source 不支持的类别当成负样本。

## 远端 18 场景 x 600 帧生产数据

远端正式数据建议使用新空目录重新生成：

```bash
conda activate habitat022
cd /path/to/bevfusion
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export REPLICA_CONFIG=/path/to/replica/replica.scene_dataset_config.json
export PRODUCTION_ROOT=/path/to/output/replica_robot_bev_v3

python -m data_generation.robot_bev.cli.generate_replica \
  --dataset "$REPLICA_CONFIG" \
  --dataset-id replica_robot_bev_v3 \
  --scenes-file data_generation/robot_bev/configs/replica_scenes.txt \
  --split-file data_generation/robot_bev/configs/replica_splits.example.json \
  --output-dir "$PRODUCTION_ROOT" \
  --num-frames 600 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh
```

预期数量：

```text
train: 8400
val:   1200
test:  1200
```

正式训练前同样执行完整 root 校验、三个 split 校验、几何诊断和
BEVFusion infos 转换。

## 中断后恢复

如果生成过程中断，并且输出目录中已有完整写入的 manifest frame，可以使用
相同命令追加 `--resume`：

```bash
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset "$REPLICA_CONFIG" \
  --dataset-id replica_robot_bev_v3 \
  --scenes-file data_generation/robot_bev/configs/replica_scenes.txt \
  --split-file data_generation/robot_bev/configs/replica_splits.example.json \
  --output-dir "$PRODUCTION_ROOT" \
  --num-frames 600 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh \
  --resume
```

恢复要求 dataset id、场景列表、split、帧数、传感器设置、navmesh 设置、
语义映射和 Habitat-Sim 版本保持一致。不要手工拼接不同 fingerprint 的输出。

## 后续接入其他数据集

后续扩充其他数据集时，不要绕过 `robot_bev_dataset v3`。新的 source adapter
只负责把自己的数据转换成统一 schema：

```text
source-specific assets
  -> data_generation/robot_bev/sources/<new_source>.py
  -> RobotBEVWriter
  -> validator
  -> robot_bev_converter.py
```

新 adapter 需要固定：

```text
1. 场景和 split 分配
2. 坐标系到 base x-forward/y-left/z-up 的转换
3. 相机 OpenCV optical 坐标
4. 六类语义映射：floor, carpet, obstacle, wall, furniture, other
5. class_validity
6. observed mask
7. generation fingerprint
```

更详细的接入要求见：

```text
data_generation/robot_bev/docs/add_new_source.md
```

## 常见检查点

开始训练前至少确认：

```text
1. multi_scene_summary.json 中 status 为 complete
2. root/train/val/test 校验均 valid: true
3. warning 已解释或处理
4. diagnostics 中几何图人工确认通过
5. bevfusion_infos_train.pkl / val.pkl / test.pkl 已生成
6. 训练配置的数据根目录指向同一个 <root>
7. 训练前冒烟检查通过
```

## RobotBEV 训练用法

当前训练适配代码包括：

```text
configs/robot_bev/README_zh.md
configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml
mmdet3d/datasets/robot_bev_dataset.py
mmdet3d/datasets/pipelines/loading.py        # LoadRobotBEVSegmentation
mmdet3d/models/heads/segm/vanilla.py         # masked focal/xent loss
tools/check_robot_bev_training.py            # 训练前冒烟检查
```

训练配置默认读取：

```text
data/replica_robot_bev_v3/bevfusion_infos_train.pkl
data/replica_robot_bev_v3/bevfusion_infos_val.pkl
data/replica_robot_bev_v3/bevfusion_infos_test.pkl
```

如果实际数据在其他目录，训练或检查时用 `dataset_root=...` 覆盖。注意
`dataset_root` 末尾要保留 `/`，因为配置中使用了字符串拼接：

```text
ann_file: ${dataset_root + "bevfusion_infos_train.pkl"}
```

```bash
docker run -it \
  --gpus all \
  --name bevfusion_dev \
  -v "$(pwd)":/workspace \
  --shm-size 16g \
  bevfusion \
  /bin/bash
```

### 训练前冒烟检查

进入训练 Docker 后先执行：

```bash
cd /path/to/bevfusion
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python tools/check_robot_bev_training.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml
```

无 CUDA 环境会只检查 dataset、dataloader 和 model 构建；有 CUDA 环境会额外跑
一个 batch 的 forward/backward。检查通过时应看到：

```text
[forward] one-batch forward/backward passed
```

如果是在宿主机直接启动 Docker，必须加 `--gpus all`，否则容器看不到 GPU：

```bash
docker run --rm --gpus all \
  -v /path/to/bevfusion:/workspace \
  -w /workspace \
  bevfusion:latest \
  python tools/check_robot_bev_training.py \
    configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml
```

只想检查数据和模型构建、不跑前后向时：

```bash
python tools/check_robot_bev_training.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --skip-forward
```

远端正式数据冒烟检查：

```bash
python tools/check_robot_bev_training.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  dataset_root=/mnt/datasets/replica_robot_bev_v3/
```

### 启动训练

正式单卡训练：

```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir work_dirs/robot_bev/camera_lidar_lss
```

远端正式数据训练时覆盖数据根目录：

```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir work_dirs/robot_bev/replica_18x600 \
  dataset_root=/mnt/datasets/replica_robot_bev_v3/
```

多卡训练时把 `-np 1` 改成 GPU 数量，例如 4 卡：

```bash
torchpack dist-run -np 4 python tools/train.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir work_dirs/robot_bev/replica_18x600 \
  dataset_root=/mnt/datasets/replica_robot_bev_v3/
```

如果从 Docker 外部直接启动训练：

```bash
docker run --rm --gpus all \
  -v /path/to/bevfusion:/workspace \
  -v /mnt/datasets/replica_robot_bev_v3:/mnt/datasets/replica_robot_bev_v3 \
  -w /workspace \
  bevfusion:latest \
  torchpack dist-run -np 1 python tools/train.py \
    configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
    --run-dir work_dirs/robot_bev/replica_18x600 \
    dataset_root=/mnt/datasets/replica_robot_bev_v3/
```

### 恢复训练

如果训练中断，可以从 checkpoint 恢复：

```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir work_dirs/robot_bev/replica_18x600 \
  dataset_root=/mnt/datasets/replica_robot_bev_v3/ \
  resume_from=work_dirs/robot_bev/replica_18x600/latest.pth
```

### 当前训练配置说明

当前配置假设：

```text
BEV 范围: x [0, 3], y [-1.5, 1.5]
BEV 分辨率: 0.02 m
BEV label shape: [6, 150, 150]
类别: floor, carpet, obstacle, wall, furniture, other
输入: camera + lidar
sweeps: 当前帧 + 最多 5 个历史点云
监督: observed_mask * class_validity * optional_per_class_supervision_mask
```

当前配置默认从官方 BEVFusion segmentation checkpoint 初始化：

```text
load_from: checkpoint/bevfusion-seg.pth
load_from_ignore_shape_mismatch: true
load_from_skip_prefixes:
  - heads.map.classifier.6
```

训练脚本会选择性加载 checkpoint：shape 完全一致的参数会加载；shape 不一致的
参数会跳过；`heads.map.classifier.6` 是最终 6 类分类层，虽然 shape 和
RobotBEV 同为 6 通道，但 nuScenes 地图类别语义和 RobotBEV 类别语义不同，
所以显式跳过并重新随机初始化。

`SwinTransformer` 的 `init_cfg` 仍然设为 `null`，避免再额外加载
`pretrained/swin_tiny_patch4_window7_224.pth`。当前初始化来源以
`checkpoint/bevfusion-seg.pth` 为准。

更详细的训练配置说明见：

```text
configs/robot_bev/README_zh.md
```
