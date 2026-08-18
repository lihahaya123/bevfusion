#!/usr/bin/env bash
set -euo pipefail

# 自采左目数据生成 RobotBEV v4 数据。
#
# 输入目录要求：
#   data/selfcollect/Left/                 左目 RGB，jpg/png
#   data/selfcollect/Depth/                uint16 毫米深度 PNG
#   data/selfcollect/Label/                二维类别 ID 标签，推荐无损 PNG
#   data/selfcollect/in.txt                fx、fy、cx、cy，每行一个值
#   data/selfcollect/CameraTrajectory.txt  timestamp tx ty tz qx qy qz qw
#
# Left、Depth、Label 使用相同文件主名进行配对，只处理左视图。
# 默认类别映射位于 configs/selfcollect_semantic_map.json；没有明确映射的
# 类别不会强行归入六类，其三维投影区域会从 BEV 有效掩膜中排除。
#
# 当前样例 Label 是有损 JPEG，因此下面显式使用 --allow-lossy-labels。
# 正式训练应将类别 ID 导出为无损 PNG，并删除该参数。
#
# 未传 --camera2base 时，生成器使用 FLOOR=4 的深度点联合估计固定相机
# 高度、横滚和俯仰；如果已有标定外参，应增加：
#   --camera2base /path/to/camera2base.txt
# 文件可以是 3x4 或 4x4，满足 p_base = camera2base @ p_camera_optical。
#
# 使用方式：在仓库根目录执行
#   bash data_generation/robot_bev/bash/generation_selfcollect.sh
# 输出目录必须为空；中断后继续生成时，在生成命令末尾增加 --resume。
#
# 默认按时间顺序周期抽帧：每连续 9 帧中的前 7 帧用于 train，第 8 帧
# 用于 val，第 9 帧用于 test。690 帧会划分为 538/76/76 帧，并分别写入
# selfcollect_001_train、selfcollect_001_val、selfcollect_001_test 三个场景。
# 输出 frame_id 会在每个场景中从 0 连续编号，原始帧号保存在帧信息的
# raw_frame_id 字段中。

export SELFCOLLECT_ROOT=/data/data3/share_data/replica_v1/data20260807/data20260807
export SELFCOLLECT_OUTPUT=/data/data3/share_data/replica_v1/data20260807/selfcollect_robot_bev

python -m data_generation.robot_bev.cli.generate_selfcollect \
  --dataset "$SELFCOLLECT_ROOT" \
  --dataset-id selfcollect_v1 \
  --scene selfcollect_001 \
  --split-mode sampled \
  --split-ratios 7 1 1 \
  --output-dir "$SELFCOLLECT_OUTPUT" \
  --semantic-map data_generation/robot_bev/configs/selfcollect_semantic_map.json \
  --allow-lossy-labels \
  --min-semantic-coverage 0.2

python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$SELFCOLLECT_OUTPUT" \
  --geometry-all-scenes \
  --geometry-frame-range 0 21 1

# 校验通过并人工确认投影方向后，可生成 BEVFusion 训练索引：
# python tools/data_converter/robot_bev_converter.py \
#   --root "$SELFCOLLECT_OUTPUT" \
#   --split all \
#   --max-sweeps 5

# 训练

docker run -it --rm  --gpus all \
   --name bev_self \
    -v "$(pwd)":/workspace \
    --user "$(id -u):$(id -g)" \
    -v /data/data3/share_data/replica_v1/data20260807:/data \
    -v /data/data3/share_data/replica_v1/lightweighting/base/original_base_bs2_ep50:/checkpoint \
    -w /workspace \
    --shm-size 32g \
    bevfusion \
    /bin/bash

CUDA_VISIBLE_DEVICES=6 torchpack dist-run -np 1 python tools/train.py \
  configs/robot_bev/seg/robotbev_selfcollect_finetune.yaml \
  --run-dir /data/selfcollect_train \
  dataset_root=/data/selfcollect_robot_bev/ \
  resume_from=null \
  load_from=/checkpoint/best_robotbev_map_iou_max_epoch_14.pth