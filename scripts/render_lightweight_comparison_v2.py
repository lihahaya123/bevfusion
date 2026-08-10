#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
渲染轻量化模型对比视频（第二组数据 v2）
布局：
  左边：原始输入图像
  右边上面：模型推理结果
  右边下面：真值标签（bev_masks）

每个模型包含两个场景：frl_apartment_5 和 office_4
为每个模型的每个场景分别生成对比视频
"""

import os
import glob
import cv2
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse


def extract_number_from_filename(filename: str) -> int:
    """从文件名中提取第一个数字用于排序"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def get_sorted_images(directory: str, scene_name: str = None) -> List[str]:
    """
    获取目录下所有png图片并按文件名中的数字排序
    
    Args:
        directory: 图像目录
        scene_name: 场景名称，用于过滤文件名（如 'frl_apartment_5' 或 'office_4'）
    """
    pattern = os.path.join(directory, "*.png")
    images = glob.glob(pattern)
    
    # 如果指定了场景名称，过滤出包含该场景名称的文件
    if scene_name:
        images = [img for img in images if scene_name in os.path.basename(img)]
    
    images.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
    return images


def get_sorted_npy(directory: str) -> List[str]:
    """获取目录下所有npy文件并按文件名中的数字排序"""
    pattern = os.path.join(directory, "*.npy")
    files = glob.glob(pattern)
    files.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
    return files


def rotate_180(image: np.ndarray) -> np.ndarray:
    """旋转图像180度"""
    return cv2.rotate(image, cv2.ROTATE_180)


def resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    """调整图像高度以匹配目标高度，保持宽高比"""
    h, w = image.shape[:2]
    target_w = int(w * (target_h / h))
    return cv2.resize(image, (target_w, target_h))


def npy_to_bev_image(npy_path: str, num_classes: int = 6) -> np.ndarray:
    """
    将npy真值文件转换为可视化图像
    
    使用与项目 mmdet3d/core/utils/visualize.py 中 visualize_map 函数
    和 data_generation/robot_bev/schema.py 中 MAP_PALETTE 保持一致的颜色。
    
    Args:
        npy_path: npy文件路径
        num_classes: 类别数量
    
    Returns:
        可视化后的图像 (H, W, 3) RGB格式
    """
    # 加载npy文件
    mask = np.load(npy_path)
    
    # 处理不同的维度情况
    if mask.ndim == 3:
        # 如果是 (C, H, W) 格式（one-hot编码）
        # 根据实际数据，npy文件形状为 (6, 150, 150)，即 (C, H, W)
        if mask.shape[0] < mask.shape[1]:  # (C, H, W) 格式
            # 先检查哪些像素是观测到的（至少有一个通道有值）
            observed = np.any(mask > 0, axis=0)  # (H, W)
            mask = np.argmax(mask, axis=0)  # 得到 (H, W)
        else:  # (H, W, C) 格式
            observed = np.any(mask > 0, axis=2)
            mask = np.argmax(mask, axis=2)
    elif mask.ndim == 1:
        # 如果是一维的，尝试推断原始尺寸
        h, w = 150, 150  # 默认尺寸
        mask = mask.reshape(h, w)
        observed = np.ones((h, w), dtype=bool)
    else:
        observed = np.ones_like(mask, dtype=bool)
    # 如果是2D (H, W)，直接使用
    
    h, w = mask.shape
    
    # 使用项目中定义的MAP_PALETTE颜色（与项目输出保持一致）
    # 来源：data_generation/robot_bev/schema.py
    # 项目使用 PIL Image.fromarray(oriented, mode="RGB") 创建RGB图像
    # 背景色使用 (240, 240, 240) 浅灰色，与 visualize_map 函数中的 background 参数一致
    result = np.full((h, w, 3), 240, dtype=np.uint8)
    
    # 注意：OpenCV使用BGR格式，需要将RGB转换为BGR
    # 项目中的MAP_PALETTE是RGB格式
    palette = [
        [216, 180, 0],      # 0: floor - 青色 (RGB: 0, 180, 216 -> BGR: 216, 180, 0)
        [255, 78, 0],       # 1: carpet - 蓝色 (RGB: 0, 78, 255 -> BGR: 255, 78, 0)
        [0, 238, 255],      # 2: wall - 黄色 (RGB: 255, 238, 0 -> BGR: 0, 238, 255)
        [31, 95, 255],      # 3: furniture - 橙色 (RGB: 255, 95, 31 -> BGR: 31, 95, 255)
        [83, 200, 0],       # 4: door - 绿色 (RGB: 0, 200, 83 -> BGR: 83, 200, 0)
        [211, 85, 186],     # 5: clutter - 紫色 (RGB: 186, 85, 211 -> BGR: 211, 85, 186)
    ]
    
    # 只对观测到的区域设置类别颜色
    for class_id in range(min(num_classes, len(palette))):
        class_mask = (mask == class_id) & observed
        result[class_mask] = palette[class_id]
    
    return result


def create_three_panel_frame(
    left_img: np.ndarray,
    right_top_img: np.ndarray,  # 推理结果
    right_bottom_img: np.ndarray  # 真值
) -> np.ndarray:
    """
    创建三面板对比帧
    左边：原始图像
    右边上面：推理结果
    右边下面：真值
    """
    # 调整推理结果和真值的高度以匹配左侧图像
    left_h = left_img.shape[0]
    panel_h = left_h // 2  # 每个面板的高度
    
    # 调整推理结果高度
    right_top_img = resize_to_height(right_top_img, panel_h)
    # 调整真值高度
    right_bottom_img = resize_to_height(right_bottom_img, panel_h)
    
    # 垂直拼接右侧两个面板
    right_panel = np.vstack([right_top_img, right_bottom_img])
    
    # 调整右侧面板宽度以匹配左侧
    right_panel = resize_to_height(right_panel, left_h)
    
    # 水平拼接
    comparison_frame = np.hstack([left_img, right_panel])
    
    return comparison_frame


def add_label(frame: np.ndarray, left_text: str, top_text: str, bottom_text: str,
              left_w: int, frame_num: int = None, font_scale=0.7) -> np.ndarray:
    """
    在视频顶部和中间添加标签
    
    布局：
      顶部标签栏（高度 label_h）：
        - 左侧标题：在左侧图像区域上方
        - 推理结果标题：在右侧面板上半部分上方
      右侧面板中间分隔处：
        - 真值标题：在右侧面板下半部分上方
    
    Args:
        frame: 三面板拼接帧（不含标签栏）
        left_text: 左侧标题
        top_text: 右上（推理结果）标题
        bottom_text: 右下（真值）标题
        left_w: 左侧图像的宽度，用于定位右侧面板起始 x 坐标
        frame_num: 帧号
        font_scale: 字体大小
    """
    h, w = frame.shape[:2]
    label_h = 30
    
    # 创建带标签的新帧（顶部增加 label_h 高度的标签栏）
    labeled_frame = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    labeled_frame[label_h:h + label_h, :] = frame
    
    # 添加顶部标签（左侧 - 原始图像）
    cv2.putText(
        labeled_frame, left_text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0), 2
    )
    
    # 添加顶部标签（右侧上面 - 推理结果）
    # 右侧面板起始 x 坐标为 left_w
    cv2.putText(
        labeled_frame, top_text,
        (left_w + 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0), 2
    )
    
    # 在右侧面板中间分隔处添加真值标签
    # y 坐标 = 标签栏高度 + 右侧上半部分高度
    mid_y = label_h + h // 2 + 20
    cv2.putText(
        labeled_frame, bottom_text,
        (left_w + 10, mid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0), 2
    )
    
    # 添加帧号（左下角）
    if frame_num is not None:
        cv2.putText(
            labeled_frame, f"Frame: {frame_num}",
            (10, h + label_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200), 1
        )
    
    return labeled_frame


def render_video(
    left_images: List[str],
    right_top_images: List[str],
    right_bottom_files: List[str],
    output_path: str,
    fps: float = 10.0,
    left_label: str = "Original",
    top_label: str = "Prediction",
    bottom_label: str = "Ground Truth"
):
    """
    渲染三面板对比视频
    
    Args:
        left_images: 左侧原始图像路径列表
        right_top_images: 右侧上面推理结果图像路径列表
        right_bottom_files: 右侧下面真值npy文件路径列表
        output_path: 输出视频路径
        fps: 帧率
        left_label: 左侧标签
        top_label: 上面标签（推理结果）
        bottom_label: 下面标签（真值）
    """
    if not left_images or not right_top_images or not right_bottom_files:
        raise ValueError("图像列表为空")
    
    # 取较短的列表
    num_frames = min(len(left_images), len(right_top_images), len(right_bottom_files))
    print(f"  处理 {num_frames} 帧...")
    
    # 读取第一帧获取尺寸
    left_frame = cv2.imread(left_images[0])
    right_top_frame = cv2.imread(right_top_images[0])
    right_bottom_frame = npy_to_bev_image(right_bottom_files[0])
    
    if left_frame is None:
        raise ValueError(f"无法读取原始图像: {left_images[0]}")
    if right_top_frame is None:
        raise ValueError(f"无法读取推理结果: {right_top_images[0]}")
    if right_bottom_frame is None:
        raise ValueError(f"无法转换真值: {right_bottom_files[0]}")
    
    # 计算输出帧尺寸
    left_h = left_frame.shape[0]
    left_w = left_frame.shape[1]
    
    # 调整推理结果和真值尺寸
    right_top_frame = resize_to_height(right_top_frame, left_h // 2)
    right_bottom_frame = resize_to_height(right_bottom_frame, left_h // 2)
    right_panel = np.vstack([right_top_frame, right_bottom_frame])
    right_panel = resize_to_height(right_panel, left_h)
    
    comp_w = left_w + right_panel.shape[1]
    comp_h = left_h + 30  # 加上标签高度
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, comp_h))
    
    for i in range(num_frames):
        # 读取图像
        left_img = cv2.imread(left_images[i])
        right_top_img = cv2.imread(right_top_images[i])
        right_bottom_img = npy_to_bev_image(right_bottom_files[i])
        
        if left_img is None or right_top_img is None or right_bottom_img is None:
            print(f"  跳过第 {i} 帧，无法读取图像")
            continue
        
        # 对推理结果和真值旋转180°
        right_top_img = rotate_180(right_top_img)
        right_bottom_img = rotate_180(right_bottom_img)
        
        # 创建三面板帧
        comp = create_three_panel_frame(left_img, right_top_img, right_bottom_img)
        
        # 添加标签（传入 left_w 用于正确定位右侧面板标题）
        comp = add_label(comp, left_label, top_label, bottom_label,
                         left_w=left_w, frame_num=i)
        
        # 写入视频
        out.write(comp)
        
        if (i + 1) % 100 == 0:
            print(f"    已处理 {i + 1}/{num_frames} 帧")
    
    out.release()
    print(f"  视频已保存: {output_path}")


def get_model_dirs(base_dir: str, exclude_dirs: set = None) -> List[str]:
    """获取模型目录列表"""
    if exclude_dirs is None:
        exclude_dirs = set()
    
    model_dirs = []
    for d in sorted(os.listdir(base_dir)):
        dir_path = os.path.join(base_dir, d)
        if os.path.isdir(dir_path) and d not in exclude_dirs:
            # 检查是否存在 result/test/map_pred 目录
            map_pred_dir = os.path.join(dir_path, "result", "test", "map_pred")
            if os.path.exists(map_pred_dir):
                model_dirs.append(d)
    
    return model_dirs


def main():
    parser = argparse.ArgumentParser(description="渲染轻量化模型对比视频（第二组数据 v2）")
    parser.add_argument(
        "--lightweighting_base", 
        type=str, 
        default=r"E:\lxx\V4\lightweighting",
        help="轻量化模型基础目录"
    )
    parser.add_argument(
        "--dataset_base", 
        type=str, 
        default=r"E:\lxx\V4\dataset",
        help="数据集基础目录"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=r"E:\lxx\V4\lightweighting\comparison_videos_v2",
        help="输出视频目录"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=10.0,
        help="视频帧率"
    )
    parser.add_argument(
        "--scenes", 
        type=str, 
        nargs='+',
        default=['frl_apartment_5', 'office_4'],
        help="要处理的场景列表"
    )
    
    args = parser.parse_args()
    
    # 定义场景配置
    scene_config = {
        'frl_apartment_5': {
            'image_dir': os.path.join(args.dataset_base, 'frl_apartment_5_mul', 'frl_apartment_5', 'images'),
            'masks_dir': os.path.join(args.dataset_base, 'frl_apartment_5_mul', 'frl_apartment_5', 'bev_masks'),
            'display_name': 'frl_apartment_5'
        },
        'office_4': {
            'image_dir': os.path.join(args.dataset_base, 'office_4_mul', 'office_4', 'images'),
            'masks_dir': os.path.join(args.dataset_base, 'office_4_mul', 'office_4', 'bev_masks'),
            'display_name': 'office_4'
        }
    }
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有模型子目录（base 和 lightcode），排除输出目录
    output_dir_abs = os.path.abspath(args.output_dir)
    sub_dirs = []
    for d in sorted(os.listdir(args.lightweighting_base)):
        sub_dir_path = os.path.join(args.lightweighting_base, d)
        if os.path.isdir(sub_dir_path):
            if os.path.abspath(sub_dir_path) == output_dir_abs:
                continue
            sub_dirs.append((d, sub_dir_path))
    
    print(f"找到 {len(sub_dirs)} 个模型子目录:")
    for name, path in sub_dirs:
        print(f"  - {name}")
    
    # 为每个场景处理
    for scene_name, config in scene_config.items():
        if scene_name not in args.scenes:
            continue
        
        print(f"\n{'='*60}")
        print(f"处理场景: {scene_name}")
        print(f"原始图像目录: {config['image_dir']}")
        print(f"真值标签目录: {config['masks_dir']}")
        print(f"{'='*60}")
        
        # 获取原始图像（原始图像文件名不包含场景名称，不需要过滤）
        left_images = get_sorted_images(config['image_dir'])
        print(f"\n找到 {len(left_images)} 张原始图像")
        
        # 获取真值标签
        gt_files = get_sorted_npy(config['masks_dir'])
        print(f"找到 {len(gt_files)} 张真值标签")
        
        if not left_images:
            print(f"  警告：未找到场景 {scene_name} 的原始图像")
            continue
        
        if not gt_files:
            print(f"  警告：未找到场景 {scene_name} 的真值标签")
            continue
        
        # 为每个子目录（base/lightcode）下的模型生成视频
        for sub_dir_name, sub_dir_path in sub_dirs:
            # 获取该子目录下的所有模型
            model_dirs = get_model_dirs(sub_dir_path)
            
            for model_dir in model_dirs:
                map_pred_dir = os.path.join(sub_dir_path, model_dir, "result", "test", "map_pred")
                right_top_images = get_sorted_images(map_pred_dir, scene_name=scene_name)
                
                if not right_top_images:
                    print(f"  [{sub_dir_name}/{model_dir}] 未找到 {scene_name} 的推理结果")
                    continue
                
                print(f"\n  [{sub_dir_name}/{model_dir}] ({len(right_top_images)} 张推理结果, {len(gt_files)} 张真值)")
                
                output_path = os.path.join(
                    args.output_dir, 
                    f"{sub_dir_name}_{model_dir}_{scene_name}_comparison.mp4"
                )
                
                try:
                    render_video(
                        left_images=left_images,
                        right_top_images=right_top_images,
                        right_bottom_files=gt_files,
                        output_path=output_path,
                        fps=args.fps,
                        left_label=f"Original ({config['display_name']})",
                        top_label=f"Prediction (rotated 180°)",
                        bottom_label=f"Ground Truth"
                    )
                except Exception as e:
                    print(f"    错误：处理 {model_dir} 时出错 - {e}")
    
    print(f"\n{'='*60}")
    print(f"所有视频生成完成！输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
