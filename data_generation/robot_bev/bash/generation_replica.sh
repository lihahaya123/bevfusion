export REPLICA_CONFIG=/data/data3/share_data/replica_v1/replicaCAD/replica.scene_dataset_config.json
export OUTPUT_ROOT=/data/data3/share_data/replica_v1/replica_robot_bev_v3_repair

# 生成数据
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset "$REPLICA_CONFIG" \
  --dataset-id replica_robot_bev_v3 \
  --scenes-file /data/data3/share_data/lixiaoxiao19/code/bevfusion/data_generation/robot_bev/configs/replica_scenes.txt \
  --split-file /data/data3/share_data/lixiaoxiao19/code/bevfusion/data_generation/robot_bev/configs//replica_splits.example.json \
  --output-dir "$OUTPUT_ROOT" \
  --num-frames 600 \
  --gpu-id 0 \
  --disable-physics \
  --turn-angle 15 \
  --recompute-navmesh

# 校验数据
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


# 转换为 BEVFusion 训练索引
python tools/data_converter/robot_bev_converter.py \
  --root "$OUTPUT_ROOT" \
  --split all \
  --max-sweeps 10
