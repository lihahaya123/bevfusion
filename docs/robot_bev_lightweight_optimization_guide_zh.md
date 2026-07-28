# Robot BEV 轻量化优化修改记录

## 1. 文档目的

本文记录 Robot BEVFusion 相机 + LiDAR 六类语义分割模型的六组轻量化方案，
用于后续在相同服务器环境下进行单变量训练和测试对比。

目标配置文件：

```text
configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml
```

分割头实现文件：

```text
mmdet3d/models/heads/segm/vanilla.py
```

本文只记录修改方法，当前基线配置和模型代码未被修改。

## 2. 当前基线

### 2.1 输入与输出

```yaml
image_size: [256, 704]
point_cloud_range: [0.0, -1.52, -0.5, 3.04, 1.52, 2.0]
voxel_size: [0.005, 0.005, 0.1]
```

当前仅启用相机和 LiDAR：

```yaml
input_modality:
  use_lidar: true
  use_camera: true
  use_radar: false
  use_map: false
  use_external: false
```

当前仅保留六类语义分割头：

```yaml
heads:
  object: null
  map:
    type: BEVSegmentationHead
    in_channels: 512
```

六个类别为：

```text
floor, carpet, wall, furniture, door, clutter
```

### 2.2 当前主要空间尺寸

```text
相机输入                         256 x 704
相机 FPN 输出特征                32 x 88
LSS 深度 bin                    39
LSS 初始 BEV                    152 x 152，2 cm
LSS 下采样后相机 BEV             76 x 76，4 cm
LiDAR 初始稀疏网格               608 x 608，5 mm
LiDAR 8 倍下采样后 BEV           76 x 76，4 cm
融合及解码器输入/输出             76 x 76
分割结果                        150 x 150，2 cm
```

当前基线模型记录：

```text
总参数量                         42,633,323
可训练参数量                     41,805,410
模型 state_dict 大小             约 162.9 MiB
```

## 3. 方案总览

| 编号 | 方案 | 仅改配置 | 需要改代码 | 是否重建数据/索引 |
|---|---|---:|---:|---:|
| E1 | LSS 直接生成 4 cm BEV | 是 | 否 | 否 |
| E2 | 分割分类后再上采样 | 否 | 是 | 否 |
| E3 | 分割通道从 512 降至 256 | 是 | 否 | 否 |
| E4 | LSS 深度步长从 0.1 m 增至 0.2 m | 是 | 否 | 否 |
| E5 | LiDAR 使用 1 cm 体素和 4 倍下采样 | 是 | 否 | 否 |
| E6 | 减少 Swin Transformer Block | 是 | 否 | 否 |

除 E2 外，其余方案均可只通过 YAML 配置完成。

## 4. E1：LSS 直接生成 4 cm BEV

### 4.1 修改目的

当前相机 LSS 先生成 2 cm、`152 x 152` 的 BEV，再通过三层 80 通道卷积
下采样到 4 cm、`76 x 76`。最终融合本身只使用 `76 x 76`，因此可以测试
直接生成 4 cm BEV，并移除这三层下采样卷积。

### 4.2 原始配置

```yaml
model:
  encoders:
    camera:
      vtransform:
        type: LSSTransform
        in_channels: 256
        out_channels: 80
        image_size: ${image_size}
        feature_size: ${[image_size[0] // 8, image_size[1] // 8]}
        xbound: [0.0, 3.04, 0.02]
        ybound: [-1.52, 1.52, 0.02]
        zbound: [-0.5, 2.0, 2.5]
        dbound: [0.1, 4.0, 0.1]
        downsample: 2
```

### 4.3 修改后配置

```yaml
model:
  encoders:
    camera:
      vtransform:
        type: LSSTransform
        in_channels: 256
        out_channels: 80
        image_size: ${image_size}
        feature_size: ${[image_size[0] // 8, image_size[1] // 8]}
        xbound: [0.0, 3.04, 0.04]
        ybound: [-1.52, 1.52, 0.04]
        zbound: [-0.5, 2.0, 2.5]
        dbound: [0.1, 4.0, 0.1]
        downsample: 1
```

### 4.4 尺寸检查

```text
x: (3.04 - 0.0) / 0.04 = 76
y: (1.52 - (-1.52)) / 0.04 = 76
```

修改后相机输出仍为 `[B, 80, 76, 76]`，与 LiDAR 输出和融合器输入一致。
以下配置保持不变：

```yaml
model:
  fuser:
    in_channels: [80, 128]

  heads:
    map:
      grid_transform:
        input_scope: [[0.0, 3.04, 0.04], [-1.52, 1.52, 0.04]]
        output_scope: [[0.0, 3.0, 0.02], [-1.5, 1.5, 0.02]]
```

### 4.5 权重加载影响

- `depthnet` 的形状不变，可以继续加载。
- 原检查点中的 `vtransform.downsample.*` 不再有对应模块，会被选择性加载器忽略。
- 其他相机主干、LiDAR、融合器、解码器和分割头权重不受影响。

### 4.6 重点检查

- `robotbev_map_iou_max`
- `robotbev_boundary_f1_50`
- `wall`、`door` 的边界 F1
- 测试平均延迟和 P95 延迟
- `cuda_peak_allocated_mb_max`

### 4.7 回退配置

```yaml
xbound: [0.0, 3.04, 0.02]
ybound: [-1.52, 1.52, 0.02]
downsample: 2
```

## 5. E2：分割分类后再上采样

### 5.1 修改目的

当前解码器输出约为 `[B, 512, 76, 76]`。分割头先将其上采样为
`[B, 512, 150, 150]`，再执行两层 `512 -> 512` 的 3x3 卷积。

当前两层卷积的理论计算量约为：

```text
2 x 150 x 150 x 512 x 512 x 3 x 3
= 106,168,320,000 MAC
```

先在 `76 x 76` 上完成分类，再只上采样 6 类 logits，可以显著减少分割头计算。

### 5.2 修改文件

```text
mmdet3d/models/heads/segm/vanilla.py
```

### 5.3 原始代码

```python
x = self.transform(x)
x = self.classifier(x)
```

### 5.4 修改后代码

```python
x = self.classifier(x)
x = self.transform(x)
```

只交换这两行的执行顺序，其他损失和输出代码保持不变。

### 5.5 输出尺寸

```text
输入                   [B, 512, 76, 76]
分类器输出              [B, 6, 76, 76]
网格变换后输出           [B, 6, 150, 150]
监督标签                [B, 6, 150, 150]
```

最终预测和监督标签尺寸不变，不需要修改数据、标签或评估代码。

### 5.6 预期变化

- 两层 512 通道卷积的理论 MAC 约减少 74%。
- `grid_sample` 的输入通道从 512 降为 6。
- 模型参数量不变。
- 原分割头检查点形状不变，可以完整加载。

### 5.7 风险

分类卷积不再直接工作在 2 cm 网格上，细边界可能受到影响。必须重点比较：

```text
map/wall/boundary_f1@0.50
map/door/boundary_f1@0.50
map/furniture/boundary_f1@0.50
robotbev_boundary_f1_50
```

### 5.8 回退代码

```python
x = self.transform(x)
x = self.classifier(x)
```

## 6. E3：分割通道从 512 降至 256

### 6.1 修改目的

当前 SECONDFPN 的两个分支各输出 256 通道，拼接后形成 512 通道。
对于只输出 6 个语义类别的任务，可以测试每个分支输出 128 通道。

### 6.2 原始配置

```yaml
model:
  decoder:
    neck:
      type: SECONDFPN
      in_channels: [128, 256]
      out_channels: [256, 256]
      upsample_strides: [1, 2]

  heads:
    map:
      type: BEVSegmentationHead
      in_channels: 512
```

### 6.3 修改后配置

```yaml
model:
  decoder:
    neck:
      type: SECONDFPN
      in_channels: [128, 256]
      out_channels: [128, 128]
      upsample_strides: [1, 2]

  heads:
    map:
      type: BEVSegmentationHead
      in_channels: 256
```

`decoder.backbone.out_channels: [128, 256]` 不需要修改。

### 6.4 预期变化

```text
分割头参数量              约 4.72M -> 1.18M
整体参数量                约 42.63M -> 38.94M
FP32 模型状态理论减少      约 14 MiB
分割头卷积理论计算量       约减少 75%
```

如果 E2 和 E3 同时生效，分割头两层主要卷积的理论 MAC 将从约
106.2G 降至约 6.8G。但第一轮实验不要同时修改，应该先分别验证。

### 6.5 权重加载影响

- `decoder.neck` 输出通道变化，其卷积和 BN 权重形状不匹配，会被跳过。
- `heads.map.classifier` 输入通道变化，分割头权重会重新初始化。
- `decoder.backbone`、融合器和两种传感器编码器可以继续加载。
- 当前配置已启用 `load_from_ignore_shape_mismatch: true`。

训练日志中应检查 `Selective checkpoint load` 信息，确认形状不匹配项符合预期。

### 6.6 保守版本

如果 256 通道精度下降明显，可以使用 384 通道折中方案：

```yaml
model:
  decoder:
    neck:
      out_channels: [192, 192]

  heads:
    map:
      in_channels: 384
```

### 6.7 回退配置

```yaml
model:
  decoder:
    neck:
      out_channels: [256, 256]

  heads:
    map:
      in_channels: 512
```

## 7. E4：LSS 深度步长从 0.1 m 增至 0.2 m

### 7.1 修改目的

当前深度范围为 0.1 m 到 4.0 m，步长为 0.1 m，共 39 个深度 bin。
将步长调整为 0.2 m 后约为 20 个 bin，可以降低 LSS lift 特征的计算量
和中间激活量。

### 7.2 原始配置

```yaml
model:
  encoders:
    camera:
      vtransform:
        dbound: [0.1, 4.0, 0.1]
```

### 7.3 修改后配置

```yaml
model:
  encoders:
    camera:
      vtransform:
        dbound: [0.1, 4.0, 0.2]
```

最大深度范围保持不变，只有深度离散步长改变。

### 7.4 中间特征变化

当前单样本、单相机 lift 特征元素数量约为：

```text
39 x 32 x 88 x 80 = 8,785,920
```

修改后约为：

```text
20 x 32 x 88 x 80 = 4,505,600
```

这部分理论元素数量约减少 49%。

### 7.5 权重加载影响

`LSSTransform.depthnet` 同时输出深度概率和相机特征：

```text
原输出通道 = D + C = 39 + 80 = 119
新输出通道 = D + C = 20 + 80 = 100
```

因此整个 `depthnet` 权重和偏置形状都会变化，并被选择性加载器跳过。
该层参数量较小，但需要通过重新训练学习深度分布和特征投影。

### 7.6 重点检查

- 远处墙面和家具 IoU
- `wall`、`door`、`furniture` 边界 F1
- 相机深度离散变粗后是否出现边缘错位
- 测试延迟和 PyTorch 峰值显存

### 7.7 中间档位

如果 0.2 m 精度下降明显，可以测试：

```yaml
dbound: [0.1, 4.0, 0.15]
```

需要注意浮点步长产生的 bin 数量应以实际 `torch.arange` 结果为准。

### 7.8 回退配置

```yaml
dbound: [0.1, 4.0, 0.1]
```

## 8. E5：LiDAR 使用 1 cm 体素和 4 倍下采样

### 8.1 修改目的

当前 LiDAR XY 体素大小为 5 mm，初始稀疏网格为 `608 x 608`，
经过三个 2 倍下采样后得到 `76 x 76`。

将体素改为 1 cm 时，初始网格变为 `304 x 304`。为了继续输出
`76 x 76`，稀疏编码器总下采样必须同步从 8 倍调整为 4 倍。

不能只修改 `voxel_size`。

### 8.2 原始顶层配置

```yaml
voxel_size: [0.005, 0.005, 0.1]
```

### 8.3 修改后顶层配置

```yaml
voxel_size: [0.01, 0.01, 0.1]
```

### 8.4 原始 LiDAR 编码器

```yaml
model:
  encoders:
    lidar:
      voxelize:
        max_num_points: 8
        point_cloud_range: ${point_cloud_range}
        voxel_size: ${voxel_size}
        max_voxels: [50000, 75000]
      backbone:
        type: SparseEncoder
        in_channels: 5
        sparse_shape: [608, 608, 25]
        output_channels: 128
        encoder_channels:
          - [16, 16, 32]
          - [32, 32, 64]
          - [64, 64, 128]
          - [128, 128]
        encoder_paddings:
          - [0, 0, 1]
          - [0, 0, 1]
          - [0, 0, [1, 1, 0]]
          - [0, 0]
        block_type: basicblock
```

### 8.5 修改后 LiDAR 编码器

```yaml
model:
  encoders:
    lidar:
      voxelize:
        max_num_points: 8
        point_cloud_range: ${point_cloud_range}
        voxel_size: ${voxel_size}
        max_voxels: [50000, 75000]
      backbone:
        type: SparseEncoder
        in_channels: 5
        sparse_shape: [304, 304, 25]
        output_channels: 128
        encoder_channels:
          - [16, 16, 32]
          - [32, 32, 64]
          - [64, 64]
        encoder_paddings:
          - [0, 0, 1]
          - [0, 0, 1]
          - [0, 0]
        block_type: basicblock
```

### 8.6 尺寸检查

```text
初始 XY 网格       3.04 / 0.01 = 304
编码器总下采样     2 x 2 = 4
最终 XY 网格       304 / 4 = 76
```

因此 LiDAR 最终输出仍为 `[B, 128, 76, 76]`，下列配置不需要修改：

```yaml
model:
  fuser:
    in_channels: [80, 128]
```

### 8.7 为什么最后一阶段使用 `[64, 64]`

当前 `SparseEncoder` 的 `basicblock` 模式会在每个非末级阶段的最后一个
block 执行 2 倍下采样。将阶段数从 4 减为 3 后，只会执行两次下采样。

末级阶段不再执行通道变化式下采样，因此使用 `[64, 64]`。随后
`conv_out` 将 64 通道转换为配置中的 `output_channels: 128`。

### 8.8 权重加载影响

- 前两个稀疏阶段中形状匹配的参数可以继续加载。
- 新第三阶段中形状匹配的 64 通道 block 可以继续加载。
- 原 128 通道第四阶段在新模型中不存在，会被忽略。
- `conv_out` 输入从 128 变为 64，权重形状不匹配，会重新初始化。

### 8.9 第一轮保持不变的参数

第一次测试 E5 时，建议保持以下参数不变：

```yaml
max_num_points: 8
max_voxels: [50000, 75000]
```

这样可以把结果变化主要归因于体素尺寸和编码器层级。确认 E5 有效后，
再单独测试：

```yaml
max_num_points: 6
max_voxels: [30000, 50000]
```

只有真实体素数量经常触及上限时，降低 `max_voxels` 才会明显减少计算；
否则它只是上限设置。

### 8.10 回退配置

恢复以下四部分：

```text
voxel_size
sparse_shape
encoder_channels
encoder_paddings
```

不能只恢复其中一项。

## 9. E6：减少 Swin Transformer Block

### 9.1 修改目的

当前相机主干为 Swin-T：

```yaml
embed_dims: 96
depths: [2, 2, 6, 2]
num_heads: [3, 6, 12, 24]
```

第三阶段包含 6 个 block，可以先减少为 4 个，在保持通道数、特征尺度、
FPN 输入形状和注意力头数不变的情况下减少参数量和计算量。

### 9.2 保守版本

```yaml
model:
  encoders:
    camera:
      backbone:
        depths: [2, 2, 4, 2]
```

### 9.3 激进版本

```yaml
model:
  encoders:
    camera:
      backbone:
        depths: [2, 2, 2, 2]
```

### 9.4 保持不变的配置

```yaml
embed_dims: 96
num_heads: [3, 6, 12, 24]
out_indices: [1, 2, 3]
```

因此 FPN 输入通道仍然为：

```yaml
in_channels: [192, 384, 768]
```

### 9.5 权重加载影响

- 保留下来的 block 形状和名称不变，可以继续加载。
- 被删除 block 的检查点参数会显示为 unexpected，并被忽略。
- FPN、LSS、LiDAR、融合器和解码器不受影响。

粗略估计：

```text
[2, 2, 4, 2] 相比基线减少约 3.5M 参数
[2, 2, 2, 2] 相比基线减少约 7M 参数
```

实际参数量以每次运行生成的 `baseline_train_metrics.jsonl` 中
`run_start.model.parameters` 为准。

### 9.6 回退配置

```yaml
depths: [2, 2, 6, 2]
```

## 10. 数据与索引

六项优化均只影响运行时网络结构、特征尺寸或体素化参数：

```text
不需要重新渲染 RGB 图像
不需要重新生成 LiDAR 点云
不需要重新生成语义标签
不需要重新生成 bevfusion_infos_*.pkl 转换索引
```

E5 的体素化发生在训练和测试 pipeline 运行期间，不写入转换索引。

## 11. 检查点使用规则

### 11.1 训练

当前训练配置包含：

```yaml
load_from: checkpoint/bevfusion-seg.pth
load_from_ignore_shape_mismatch: true
```

因此训练时会：

```text
加载名称和形状均匹配的参数
跳过形状变化的参数
跳过新模型中不存在的旧参数
随机初始化新增或形状变化的参数
```

每次启动训练后，都应检查日志中的：

```text
Selective checkpoint load
loaded
skipped_shape
unexpected
missing
```

### 11.2 测试

优化模型必须使用：

```text
与该优化模型结构匹配的配置文件
该优化实验自己训练产生的检查点
```

不要直接使用基线结构的 `latest.pth` 测试 E3、E4、E5 或 E6 模型。

## 12. 推荐实验方式

### 12.1 第一阶段：单变量消融

每个实验都从基线配置单独派生，不叠加其他修改：

| 实验名 | 相对基线的唯一修改 |
|---|---|
| B0 | 无修改 |
| A1 | 仅 E1 |
| A2 | 仅 E2 |
| A3 | 仅 E3 |
| A4 | 仅 E4 |
| A5 | 仅 E5 |
| A6 | 仅 E6 保守版 |

这样可以分别判断每项修改的精度损失和收益。

### 12.2 第二阶段：逐项组合

选择单变量实验中通过验收的方案，再按以下顺序逐步叠加：

```text
C1 = E1
C2 = E1 + E2
C3 = E1 + E2 + E4
C4 = E1 + E2 + E4 + E3
C5 = E1 + E2 + E4 + E3 + E5
C6 = E1 + E2 + E4 + E3 + E5 + E6
```

如果暂时不修改代码，则使用纯配置路线：

```text
E1 -> E4 -> E3 -> E5 -> E6
```

### 12.3 远端训练参数保持一致

每组实验保持：

```text
单卡训练
samples_per_gpu = 2
workers_per_gpu = 2
optimizer.lr = 5e-5
max_epochs = 30
相同数据集与划分
相同随机种子
相同早停配置
```

示例：

```bash
python tools/train.py \
  configs/robot_bev/seg/<experiment_config>.yaml \
  --run-dir /output/lightweight/<experiment_name> \
  dataset_root=/data/ \
  data.samples_per_gpu=2 \
  data.workers_per_gpu=2 \
  optimizer.lr=5e-5 \
  max_epochs=30
```

实际远端仍可由现有 `mpirun` 和 TorchPack 启动器包装该命令。

## 13. 每次实验需要比较的指标

### 13.1 精度

```text
robotbev_map_iou_max
robotbev_map_iou_50
robotbev_map_f1_50
robotbev_boundary_f1_50
六类 map/<class>/iou@max
六类 map/<class>/boundary_f1@0.50
best_epoch
```

不能只查看均值。`door`、`wall` 和 `furniture` 对空间分辨率变化更敏感，
应单独检查。

### 13.2 速度

```text
train_samples_per_second
train_ms_per_sample
train_epoch_seconds
model_latency.mean
model_latency.p50
model_latency.p95
end_to_end_ms_per_sample
```

### 13.3 模型和内存

```text
model.parameters
model.trainable_parameters
model.model_state_size_mb
cuda_peak_allocated_mb_max
cuda_peak_reserved_mb_max
peak_rss_mb
```

当前 PyTorch 指标不能完整反映 `nvidia-smi` 进程总显存，但在相同环境下
仍可用于比较 PyTorch 管理的模型显存变化。

### 13.4 对比计算

```text
IoU 变化 = 优化模型 best IoU - 基线 best IoU
推理加速比 = 基线平均延迟 / 优化模型平均延迟
吞吐提升率 = 优化吞吐 / 基线吞吐 - 1
参数减少率 = 1 - 优化参数量 / 基线参数量
PyTorch 显存减少率 = 1 - 优化峰值显存 / 基线峰值显存
```

早停可能导致不同实验训练 epoch 数不同，因此不能只比较整个训练任务的
总时间，应优先比较单 epoch 时间、吞吐量、最佳精度和达到目标精度所需时间。

## 14. 修改前检查清单

每次开始实验前确认：

```text
[ ] 从基线或明确记录的组合配置开始
[ ] 本次只修改计划中的一组参数
[ ] 相机和 LiDAR 最终 BEV 都是 76 x 76
[ ] fuser.in_channels 与两分支输出通道一致
[ ] heads.map.in_channels 等于 SECONDFPN 输出通道之和
[ ] 使用独立 run-dir，避免覆盖基线结果
[ ] 检查选择性检查点加载日志
[ ] 保存最终展开后的 configs.yaml
[ ] 使用对应实验检查点进行测试
[ ] 同时比较总体 IoU 和各类别边界指标
```

## 15. 当前建议

优先执行 E1，因为它：

```text
只修改配置
保持最终融合尺寸不变
不改变深度 bin
大部分预训练权重可以继续加载
不需要重新生成数据或索引
```

完成 E1 的完整训练和测试后，再决定是否进入 E2 或 E4。不要第一次就将
E1、E3、E4 和 E5 同时修改，否则难以定位精度变化来源。
