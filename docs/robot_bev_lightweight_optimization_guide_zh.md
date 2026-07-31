# Robot BEV 轻量化优化修改清单

基线配置：

```text
configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml
```

每个阶段均单独从当前基线派生，不与其他阶段叠加。

## A1：LSS 直接生成 4 cm BEV

修改相机 `vtransform`：

```yaml
xbound: [0.0, 3.04, 0.04]
ybound: [-1.52, 1.52, 0.04]
downsample: 1
```

其余配置保持不变，LSS 输出仍为 `76 × 76`。

## A2：分割头先分类、后上采样

修改：

```text
mmdet3d/models/heads/segm/vanilla.py
```

将：

```python
x = self.transform(x)
x = self.classifier(x)
```

改为：

```python
x = self.classifier(x)
x = self.transform(x)
```

最终输出仍为 `[B, 6, 150, 150]`。

## A3：LSS depth bin 步长从 0.05 m 增至 0.1 m

修改相机 `vtransform`：

```yaml
dbound: [0.1, 5.0, 0.1]
```

深度 bin 数由 98 降至 49。

## A4：分割通道从 512 降至 256

修改解码器 neck：

```yaml
model:
  decoder:
    neck:
      out_channels: [128, 128]
```

修改分割头：

```yaml
model:
  heads:
    map:
      in_channels: 256
```

`decoder.backbone.out_channels` 保持 `[128, 256]`。

## A5：LiDAR 使用 1 cm 体素和 4 倍下采样

修改体素尺寸：

```yaml
voxel_size: [0.01, 0.01, 0.1]
```

修改 LiDAR 稀疏编码器：

```yaml
model:
  encoders:
    lidar:
      backbone:
        sparse_shape: [304, 304, 25]
        encoder_channels:
          - [16, 16, 32]
          - [32, 32, 64]
          - [64, 64]
        encoder_paddings:
          - [0, 0, 1]
          - [0, 0, 1]
          - [0, 0]
```

`output_channels: 128`、`max_num_points` 和 `max_voxels` 保持基线值不变。最终 LiDAR 输出仍为 `76 × 76`。

## A6：减少 Swin Transformer Blocks

修改相机 backbone：

```yaml
model:
  encoders:
    camera:
      backbone:
        depths: [2, 2, 4, 2]
```

`embed_dims`、`num_heads`、`out_indices` 和 FPN 输入通道保持不变。

## B1：LiDAR Pillar + 2D BEV 编码器

将 3D voxel 和 SparseConv3D 替换为 XY Pillar、高度统计特征和轻量 2D BEV 编码器。

## B2：轻量分割与边界细化头

在低分辨率 BEV 上完成分类，再上采样 6 类 logits，并用少量通道进行边界细化。

## B3：门控加法融合

将相机和 LiDAR 特征投影到相同通道数，通过门控加法替代通道拼接和重卷积。

## B4：单尺度轻量 BEV Decoder

用少量 Depthwise Residual Blocks 替换 SECOND + SECONDFPN 多尺度解码结构。

## B5：稀疏 LSS 深度投影

每个像素只保留少量高概率深度候选，再投影到 BEV，减少完整 depth-bin lift 计算。

## B6：轻量相机骨干

用移动端 CNN 或轻量 CNN-Transformer 混合骨干替换 Swin Transformer。

## B7：历史 BEV 特征缓存

缓存并对齐历史帧 BEV 特征，替代历史点云的重复加载、体素化和编码。
