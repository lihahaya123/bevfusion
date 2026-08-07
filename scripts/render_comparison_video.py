#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
渲染对比视频脚本
左边：原始图像 (Left目录)
右边：推理结果图像 (bx50下各模型目录)，旋转180度后展示

为bx50下的15个模型分别生成15个对比视频
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse


import re


def extract_number_from_filename(filename: str) -> int:
    """从文件名中提取第一个数字用于排序"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def get_sorted_images(directory: str) -> List[str]:
    """获取目录下所有png图片并按文件名中的数字排序"""
    pattern = os.path.join(directory, "*.png")
    images = glob.glob(pattern)
    images.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
    return images


def get_left_images(directory: str) -> List[str]:
    """获取Left目录下所有jpg图片并按文件名中的数字排序"""
    pattern = os.path.join(directory, "*.jpg")
    images = glob.glob(pattern)
    images.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
    return images


def rotate_180(image: np.ndarray) -> np.ndarray:
    """旋转图像180度"""
    return cv2.rotate(image, cv2.ROTATE_180)


def resize_to_match(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """调整图像大小以匹配目标尺寸"""
    return cv2.resize(img, (target_shape[1], target_shape[0]))


def create_comparison_frame(
    left_img: np.ndarray,
    right_img: np.ndarray
) -> np.ndarray:
    """
    创建左右对比帧
    左边：原始图像
    右边：推理结果（旋转180度）
    """
    # 旋转推理结果180度
    right_img_rotated = rotate_180(right_img)
    
    # 调整推理结果高度以匹配原始图像高度，保持宽高比
    left_h = left_img.shape[0]
    right_h, right_w = right_img_rotated.shape[:2]
    left_w = left_img.shape[1]
    
    # 按原始图像高度缩放推理结果
    target_right_h = left_h
    target_right_w = int(right_w * (target_right_h / right_h))
    right_img_rotated = cv2.resize(right_img_rotated, (target_right_w, target_right_h))
    
    # 水平拼接
    comparison_frame = np.hstack([left_img, right_img_rotated])
    
    return comparison_frame


def add_label(frame: np.ndarray, left_text: str, right_text: str, font_scale=1.0) -> np.ndarray:
    """在视频顶部添加标签"""
    h, w = frame.shape[:2]
    label_h = 40
    
    # 创建带标签的新帧
    labeled_frame = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    labeled_frame[label_h:h + label_h, :] = frame
    
    # 添加左标签
    cv2.putText(
        labeled_frame, left_text, 
        (20, 25), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (0, 255, 0), 2
    )
    
    # 添加右标签
    cv2.putText(
        labeled_frame, right_text, 
        (w // 2 + 20, 25), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (255, 0, 0), 2
    )
    
    return labeled_frame


def render_video(
    left_images: List[str],
    right_images: List[str],
    output_path: str,
    fps: float = 10.0,
    left_label: str = "Original",
    right_label: str = "Prediction"
):
    """
    渲染对比视频
    
    Args:
        left_images: 左侧原始图像路径列表
        right_images: 右侧推理结果图像路径列表
        output_path: 输出视频路径
        fps: 帧率
        left_label: 左侧标签
        right_label: 右侧标签
    """
    if not left_images or not right_images:
        raise ValueError("图像列表为空")
    
    # 取较短的列表
    num_frames = min(len(left_images), len(right_images))
    print(f"处理 {num_frames} 帧...")
    
    # 读取第一帧获取尺寸
    left_frame = cv2.imread(left_images[0])
    right_frame = cv2.imread(right_images[0])
    
    if left_frame is None or right_frame is None:
        raise ValueError("无法读取图像")
    
    # 旋转推理结果并按左侧图像高度缩放
    right_frame_rotated = rotate_180(right_frame)
    left_h = left_frame.shape[0]
    right_h, right_w = right_frame_rotated.shape[:2]
    target_right_w = int(right_w * (left_h / right_h))
    right_frame_rotated = cv2.resize(right_frame_rotated, (target_right_w, left_h))
    
    # 获取输出帧尺寸
    comp_w = left_frame.shape[1] + right_frame_rotated.shape[1]
    comp_h = left_frame.shape[0] + 40  # 加上标签高度
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, comp_h))
    
    for i in range(num_frames):
        # 读取图像
        left_img = cv2.imread(left_images[i])
        right_img = cv2.imread(right_images[i])
        
        if left_img is None or right_img is None:
            print(f"跳过第 {i} 帧，无法读取图像")
            continue
        
        # 创建对比帧
        comp = create_comparison_frame(left_img, right_img)
        
        # 添加标签
        comp = add_label(comp, left_label, right_label)
        
        # 写入视频
        out.write(comp)
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{num_frames} 帧")
    
    out.release()
    print(f"视频已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="渲染原始图像与推理结果的对比视频")
    parser.add_argument(
        "--left_dir", 
        type=str, 
        default=r"E:\lxx\V4\selfcollect\data20260807\Left",
        help="原始图像目录路径"
    )
    parser.add_argument(
        "--right_base_dir", 
        type=str, 
        default=r"E:\lxx\V4\selfcollect\infer\bx50",
        help="推理结果基础目录路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=r"E:\lxx\V4\selfcollect\infer\bx50\comparison_videos",
        help="输出视频目录路径"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=10.0,
        help="视频帧率"
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="map_pred",
        help="推理结果子目录名（相对于每个模型目录的result/test/）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有模型目录（排除输出目录）
    output_dir_abs = os.path.abspath(args.output_dir)
    model_dirs = []
    for d in sorted(os.listdir(args.right_base_dir)):
        dir_path = os.path.join(args.right_base_dir, d)
        if os.path.isdir(dir_path):
            # 排除输出目录
            if os.path.abspath(dir_path) == output_dir_abs:
                continue
            model_dirs.append(d)
    
    print(f"找到 {len(model_dirs)} 个模型目录:")
    for d in model_dirs:
        print(f"  - {d}")
    
    # 获取原始图像列表
    left_images = get_left_images(args.left_dir)
    print(f"\n找到 {len(left_images)} 张原始图像")
    
    if not left_images:
        print("错误：未找到原始图像")
        return
    
    # 为每个模型生成对比视频
    for model_dir in model_dirs:
        right_dir = os.path.join(
            args.right_base_dir, model_dir, "result", "test", args.result_subdir
        )
        
        if not os.path.exists(right_dir):
            print(f"跳过 {model_dir}：目录不存在 {right_dir}")
            continue
        
        right_images = get_sorted_images(right_dir)
        print(f"\n处理模型: {model_dir} ({len(right_images)} 张推理结果)")
        
        if not right_images:
            print(f"  跳过：未找到推理结果图像")
            continue
        
        output_path = os.path.join(args.output_dir, f"{model_dir}_comparison.mp4")
        
        try:
            render_video(
                left_images=left_images,
                right_images=right_images,
                output_path=output_path,
                fps=args.fps,
                left_label="Original (Left)",
                right_label=f"{model_dir} (Right, rotated 180°)"
            )
        except Exception as e:
            print(f"  错误：处理 {model_dir} 时出错 - {e}")
    
    print("\n所有视频生成完成！")


if __name__ == "__main__":
    main()
