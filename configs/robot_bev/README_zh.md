# RobotBEV 训练配置说明

当前 RobotBEV 训练入口配置：

```text
configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml
```

它读取 `robot_bev_dataset v4` 转换后的 BEVFusion 索引：

```text
data/replica_robot_bev_v4/bevfusion_infos_train.pkl
data/replica_robot_bev_v4/bevfusion_infos_val.pkl
data/replica_robot_bev_v4/bevfusion_infos_test.pkl
```

核心适配点：

```text
RobotBEVDataset
  -> LoadRobotBEVSegmentation
  -> gt_masks_bev + gt_supervision_mask_bev
  -> BEVFusion map head
  -> masked focal loss
```

`gt_supervision_mask_bev` 的计算规则为：

```text
observed_mask[None, :, :]
  * class_validity[:, None, None]
  * optional_per_class_supervision_mask
```

也就是说，未观测区域和 source 不支持的类别不会作为负样本参与损失。

## 训练前冒烟检查

在训练 Docker 中先执行：

```bash
cd /path/to/bevfusion
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python tools/check_robot_bev_training.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml
```

无 CUDA 的环境会完成 dataset、dataloader、model 构建检查，然后跳过 forward。
有 CUDA 的环境会额外跑一个 batch 的 forward/backward。

如果只想检查加载链路，不跑前后向：

```bash
python tools/check_robot_bev_training.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --skip-forward
```

如果数据目录不在默认位置，可以临时覆盖：

```bash
python tools/check_robot_bev_training.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  dataset_root=/mnt/datasets/replica_robot_bev_v4/
```

注意末尾 `/` 要保留，因为配置里使用了：

```text
ann_file: ${dataset_root + "bevfusion_infos_train.pkl"}
```

## 启动训练

单机训练：

```bash
cd /workspace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

torchpack dist-run -np 1 python tools/train.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir work_dirs/robot_bev/camera_lidar_lss
```

远端正式数据训练时覆盖数据根目录：

```bash
cd /workspace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

torchpack dist-run -np 1 python tools/train.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir work_dirs/robot_bev/replica_18x600 \
  dataset_root=/mnt/datasets/replica_robot_bev_v4/
```

多卡时把 `-np 1` 改成 GPU 数量，例如：

```bash
torchpack dist-run -np 4 python tools/train.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir work_dirs/robot_bev/replica_18x600 \
  dataset_root=/mnt/datasets/replica_robot_bev_v4/
```

## 基线指标记录

当前配置默认启用：

```yaml
baseline_metrics:
  enabled: true
  train_output: baseline_train_metrics.jsonl
  inference_warmup_batches: 5
```

训练时会在 `--run-dir` 下追加写入
`baseline_train_metrics.jsonl`。文件包含运行环境和模型规模，以及每个 epoch
的平均 loss、验证集指标、训练吞吐、训练耗时、进程内存和训练峰值显存。

测试时会在 checkpoint 所在目录生成
`baseline_test_<checkpoint>_<timestamp>.json`。也可以显式指定路径：

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  work_dirs/robot_bev/camera_lidar_lss/latest.pth \
  --eval map \
  --baseline-metrics-out work_dirs/robot_bev/camera_lidar_lss/baseline_test.json
```

测试基线文件包含固定阈值 `0.50` 下的 IoU、Precision、Recall、F1，以及
允许 1 个 BEV 像素误差的 Boundary F1；还包含参数量、checkpoint 大小、
端到端吞吐、模型延迟 P50/P95/P99、进程内存和峰值显存。当前 BEV
分辨率为 2 cm，因此边界容差对应 2 cm。模型延迟统计会排除配置指定的前
5 个 warmup batch，端到端吞吐仍覆盖完整测试流程。

## 早停

当前训练配置默认启用验证集早停：

```yaml
early_stopping:
  enabled: true
  monitor: robotbev_map_iou_max
  rule: greater
  patience: 5
  min_delta: 0.0001
  start_epoch: 1
  state_file: early_stopping_state.json
```

每个 epoch 验证后检查 `robotbev_map_iou_max`。指标提升超过 `0.0001` 时更新
best 并清零计数；连续 5 次验证没有达到该提升幅度时结束训练。监控指标与
`evaluation.save_best` 相同，因此早停判断和 best checkpoint 选择保持一致。

状态保存在 `--run-dir/early_stopping_state.json`，记录最佳分数、最佳
epoch、连续未提升次数和停止 epoch。同一个 `--run-dir` 使用
`resume_from=...` 恢复时会继续原有计数。临时关闭早停可在训练命令末尾添加：

```bash
early_stopping.enabled=false
```

## Checkpoint 保存逻辑

当前配置：

```yaml
checkpoint_config:
  interval: 1
  max_keep_ckpts: 1

evaluation:
  interval: 1
  save_best: robotbev_map_iou_max
  rule: greater
```

训练过程中会保存：

```text
latest.pth                                  # 最新 checkpoint，通常是软链接
epoch_<N>.pth                               # 第 N 个 epoch 的 checkpoint
best_robotbev_map_iou_max_epoch_<N>.pth     # 验证集 robotbev_map_iou_max 最好的 checkpoint
```

其中 `robotbev_map_iou_max` 与日志里的 `map/mean/iou@max` 数值相同，是用于
保存 best checkpoint 的文件名安全别名。

## 当前配置假设

```text
BEV 标签范围: x [0, 3], y [-1.5, 1.5], z [-0.5, 2.0]
BEV 分辨率: 0.02 m
BEV label shape: [6, 150, 150]
类别: floor, carpet, wall, furniture, door, clutter
语义标签来源: semantic-depth 点投影，按上述 x/y/z 范围过滤，不混入 navmesh 可通行性
输入: camera + lidar
sweeps: 当前帧 + 最多 5 个历史点云
```

当前配置默认从官方 BEVFusion segmentation checkpoint 初始化：

```text
load_from: checkpoint/bevfusion-seg.pth
load_from_ignore_shape_mismatch: true
load_from_skip_prefixes:
  - heads.map.classifier.6
```

训练脚本会选择性加载 checkpoint：shape 完全一致的参数会加载；shape 不一致的
参数会跳过；最终 6 类分类层 `heads.map.classifier.6` 虽然 shape 对得上，
但 nuScenes 地图类别和 RobotBEV 类别语义不同，因此显式跳过并重新初始化。

`SwinTransformer` 的 `init_cfg` 仍然设为 `null`，避免再额外加载
`pretrained/swin_tiny_patch4_window7_224.pth`。当前初始化来源以
`checkpoint/bevfusion-seg.pth` 为准。
