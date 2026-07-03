# 代码框架
```
Camera 分支是三段式流水线：图像 → backbone提取2D特征 → neck融合多尺度特征 → vtransform投影到BEV

LiDAR 分支是两段式：点云 → voxelize体素化 → backbone(SparseEncoder)提取BEV特征
    两种体素化的区别：
    Voxelization：硬体素化，每个体素最多 max_num_points 个点，超出丢弃，输出固定大小张量
    DynamicScatter：动态体素化，不限制点数，所有点都保留，通过散射聚合
```