# mmdet3d/models 模块说明

## 整体架构

```
输入数据(相机/激光雷达/雷达)
        │
        ▼
┌─────────────────┐
│   backbones/     │  ← 骨干网络，特征提取
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────┐
│ vtransforms/ │ │   necks/      │  ← 视图变换 / 颈部特征增强
└────┬───┘ └──────┬──────┘
     │            │
     ▼            ▼
┌─────────────────┐
│    fusers/       │  ← 多模态BEV融合
└────────┬────────┘
         ▼
┌─────────────────┐
│     heads/       │  ← 检测/分割任务头
└────────┬────────┘
         ▼
┌─────────────────┐
│    losses/       │  ← 损失函数
└─────────────────┘

整个流程由 fusion_models/ 统一编排
builder.py 提供注册表与构建工厂
utils/ 提供Transformer等工具组件
```

---

## 各文件夹详细说明

### 1. backbones/ — 骨干网络（特征提取）

负责从原始传感器数据中提取特征，是整个网络的"特征提取器"。

| 文件 | 类/模型 | 作用 |
|------|---------|------|
| `resnet.py` | ResNet | ResNet系列，用于**相机图像**特征提取 |
| `vovnet.py` | VoVNet | 另一种相机图像骨干网络 |
| `second.py` | SECOND | 用于**激光雷达体素**特征提取 |
| `sparse_encoder.py` | SparseEncoder | 稀疏卷积编码器，处理稀疏体素特征 |
| `pillar_encoder.py` | PillarEncoder | PointPillars柱体编码器，将点云转为伪图像 |
| `dla.py` | DLA | Deep Layer Aggregation骨干网络 |
| `radar_encoder.py` | RadarEncoder | 雷达点云编码器 |

### 2. vtransforms/ — 视图变换（View Transform）

将 **2D 相机图像特征** 投影到 **3D/BEV 空间**，使相机特征能与激光雷达 BEV 特征对齐融合。

| 文件 | 类/模型 | 作用 |
|------|---------|------|
| `base.py` | BaseTransform | 视图变换基类 |
| `lss.py` | LSSViewTransform | LSS（Lift-Splat-Shoot）视图变换 |
| `depth_lss.py` | DepthLSSViewTransform | 带深度估计的LSS变换 |
| `aware_bevdepth.py` | BEVDepthAware | BEVDepth感知型视图变换 |

### 3. necks/ — 颈部网络（特征增强）

连接骨干与检测头，做特征金字塔、多尺度融合等。

| 文件 | 类/模型 | 作用 |
|------|---------|------|
| `second.py` | SECONDFPN | SECOND FPN，激光雷达分支常用 |
| `detectron_fpn.py` | DetectronFPN | Detectron风格的FPN |
| `lss.py` | LSSFPN | LSS颈部，配合视图变换使用 |
| `generalized_lss.py` | GeneralizedLSSFPN | 泛化版LSS颈部 |

### 4. fusers/ — 多模态融合器

将**相机BEV特征**与**激光雷达BEV特征**在BEV空间进行融合。

| 文件 | 类/模型 | 作用 |
|------|---------|------|
| `add.py` | AddFuser | 逐元素相加融合 |
| `conv.py` | ConvFuser | 卷积融合（拼接后卷积） |

### 5. heads/ — 任务头

输出最终预测结果，分检测和分割两类。

| 子目录/文件 | 类/模型 | 作用 |
|------|---------|------|
| `bbox/centerpoint.py` | CenterHead | CenterPoint检测头（基于中心点热力图） |
| `bbox/transfusion.py` | TransFusionHead | TransFusion检测头（Transformer融合） |
| `segm/vanilla.py` | VanillaBEVSegHead | BEV语义分割头 |

### 6. losses/ — 损失函数

提供训练监督所需的损失函数，复用mmdet的损失实现。

| 损失函数 | 作用 |
|----------|------|
| `FocalLoss` | 用于分类任务，解决类别不平衡 |
| `SmoothL1Loss` | 用于回归任务（bbox） |
| `binary_cross_entropy` | 二值交叉熵，用于分割任务 |

### 7. fusion_models/ — 融合模型（整体编排）

**顶层模型容器**，把上述各模块组装成完整可训练/推理的模型。

| 文件 | 类/模型 | 作用 |
|------|---------|------|
| `base.py` | Base3DFusionModel | 融合模型基类，处理损失解析等通用逻辑 |
| `bevfusion.py` | BEVFusion | **BEVFusion主模型**，编排相机分支、激光雷达分支、融合器、检测/分割头 |

### 8. utils/ — 工具模块

| 文件 | 作用 |
|------|------|
| `transformer.py` | Transformer相关组件（供TransFusion等使用） |
| `flops_counter.py` | FLOPs计算工具 |

### 9. builder.py — 构建器

注册表与工厂函数，定义三个自定义注册表并提供构建函数：

| 注册表/函数 | 说明 |
|-------------|------|
| `FUSIONMODELS` | 融合模型注册表 |
| `VTRANSFORMS` | 视图变换注册表 |
| `FUSERS` | 融合器注册表 |
| `build_model()` | 根据配置构建完整模型 |
| `build_vtransform()` | 构建视图变换模块 |
| `build_fuser()` | 构建融合器模块 |

---

## BEVFusion 数据流总结

```
相机图像 ──→ backbone(ResNet/VoVNet) ──→ vtransform(LSS) ──→ 相机BEV特征 ──┐
                                                                          ├─→ fuser ──→ heads ──→ 检测/分割结果
激光雷达 ──→ backbone(SparseEncoder) ──→ neck(SECOND FPN) ──→ LiDAR BEV特征 ─┘
```

1. **相机分支**：图像 → backbone提取2D特征 → vtransform投影到BEV空间
2. **激光雷达分支**：点云 → voxelize体素化 → backbone(SparseEncoder)提取BEV特征 → neck(FPN)增强
3. **融合**：两路BEV特征经fuser融合
4. **输出**：融合特征送入heads，输出3D检测或BEV分割结果
5. **训练**：losses计算损失，反向传播更新参数

整个流程由 `fusion_models/bevfusion.py` 中的 BEVFusion 类统一编排调度。
