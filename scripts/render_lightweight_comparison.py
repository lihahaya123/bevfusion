#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
渲染轻量化模型对比视频（第二组数据）
左边：原始图像 (E:\lxx\V4\dataset\{scene}_mul\{scene}\images)
右边：推理结果图像 (E:\lxx\V4\lightweighting\{base|lightcode}\{model}\result\test\map_pred)

每个模型包含两个场景：frl_apartment_5 和 office_4
为每个模型的每个场景分别生成对比视频

目录结构：
- lightweighting/base/ 包含 7 个模型
- lightweighting/lightcode/ 包含 8 个模型
- 共 15 个模型
"""

import os
import glob
import cv2
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


def extract_number_from_filename(filename: str) -> int:
    """从文件名中提取第一个数字用于排序"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def get_sorted_images(directory: str, scene_prefix: str = None) -> List[str]:
    """
    获取目录下所有png图片并按文件名中的数字排序
    scene_prefix: 可选，只获取包含该前缀的文件（如 'frl_apartment_5' 或 'office_4'）
    """
    pattern = os.path.join(directory, "*.png")
    images = glob.glob(pattern)
    
    if scene_prefix:
        # 过滤出包含场景前缀的文件
        images = [f for f in images if scene_prefix in os.path.basename(f)]
    
    images.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
    return images


def rotate_180(image: np.ndarray) -> np.ndarray:
    """旋转图像180度"""
    return cv2.rotate(image, cv2.ROTATE_180)


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
    
    # 按原始图像高度缩放推理结果
    target_right_h = left_h
    target_right_w = int(right_w * (target_right_h / right_h))
    right_img_rotated = cv2.resize(right_img_rotated, (target_right_w, target_right_h))
    
    # 水平拼接
    comparison_frame = np.hstack([left_img, right_img_rotated])
    
    return comparison_frame


def add_label(frame: np.ndarray, left_text: str, right_text: str, frame_num: int = None, font_scale=0.8) -> np.ndarray:
    """在视频顶部添加标签"""
    h, w = frame.shape[:2]
    label_h = 50
    
    # 创建带标签的新帧
    labeled_frame = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    labeled_frame[label_h:h + label_h, :] = frame
    
    # 添加左标签
    cv2.putText(
        labeled_frame, left_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (0, 255, 0), 2
    )
    
    # 添加右标签
    cv2.putText(
        labeled_frame, right_text, 
        (w // 2 + 10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (255, 0, 0), 2
    )
    
    # 添加帧号
    if frame_num is not None:
        cv2.putText(
            labeled_frame, f"Frame: {frame_num}", 
            (10, h + label_h - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (200, 200, 200), 1
        )
    
    return labeled_frame


def render_video(
    left_images: List[str],
    right_images: List[str],
    output_path: str,
    fps: float = 10.0,
    left_label: str = "Original",
    right_label: str = "Prediction",
    scene_name: str = ""
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
        scene_name: 场景名称（用于标签）
    """
    if not left_images or not right_images:
        raise ValueError("图像列表为空")
    
    # 取较短的列表
    num_frames = min(len(left_images), len(right_images))
    print(f"  处理 {num_frames} 帧...")
    
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
    comp_h = left_frame.shape[0] + 50  # 加上标签高度
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, comp_h))
    
    for i in range(num_frames):
        # 读取图像
        left_img = cv2.imread(left_images[i])
        right_img = cv2.imread(right_images[i])
        
        if left_img is None or right_img is None:
            print(f"  跳过第 {i} 帧，无法读取图像")
            continue
        
        # 创建对比帧
        comp = create_comparison_frame(left_img, right_img)
        
        # 添加标签
        comp = add_label(comp, left_label, right_label, frame_num=i)
        
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
    parser = argparse.ArgumentParser(description="渲染轻量化模型对比视频（第二组数据）")
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
        default=r"E:\lxx\V4\lightweighting\comparison_videos",
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
            'prefix': 'frl_apartment_5',
            'display_name': 'frl_apartment_5'
        },
        'office_4': {
            'image_dir': os.path.join(args.dataset_base, 'office_4_mul', 'office_4', 'images'),
            'prefix': 'office_4',
            'display_name': 'office_4'
        }
    }
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有模型子目录（base 和 lightcode）
    sub_dirs = []
    for d in sorted(os.listdir(args.lightweighting_base)):
        sub_dir_path = os.path.join(args.lightweighting_base, d)
        if os.path.isdir(sub_dir_path):
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
        print(f"推理结果前缀: {config['prefix']}")
        print(f"{'='*60}")
        
        # 获取原始图像
        left_images = get_sorted_images(config['image_dir'])
        print(f"\n找到 {len(left_images)} 张原始图像")
        
        if not left_images:
            print(f"  警告：未找到场景 {scene_name} 的原始图像")
            continue
        
        # 为每个子目录（base/lightcode）下的模型生成视频
        for sub_dir_name, sub_dir_path in sub_dirs:
            # 获取该子目录下的所有模型
            model_dirs = get_model_dirs(sub_dir_path)
            
            for model_dir in model_dirs:
                map_pred_dir = os.path.join(sub_dir_path, model_dir, "result", "test", "map_pred")
                right_images = get_sorted_images(map_pred_dir, scene_prefix=config['prefix'])
                
                if not right_images:
                    print(f"  [{sub_dir_name}/{model_dir}] 未找到 {scene_name} 的推理结果")
                    continue
                
                print(f"\n  [{sub_dir_name}/{model_dir}] ({len(right_images)} 张推理结果)")
                
                output_path = os.path.join(
                    args.output_dir, 
                    f"{sub_dir_name}_{model_dir}_{scene_name}_comparison.mp4"
                )
                
                try:
                    render_video(
                        left_images=left_images,
                        right_images=right_images,
                        output_path=output_path,
                        fps=args.fps,
                        left_label=f"Original ({config['display_name']})",
                        right_label=f"{sub_dir_name}/{model_dir} (rotated 180°)",
                        scene_name=scene_name
                    )
                except Exception as e:
                    print(f"    错误：处理 {model_dir} 时出错 - {e}")
    
    print(f"\n{'='*60}")
    print(f"所有视频生成完成！输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
