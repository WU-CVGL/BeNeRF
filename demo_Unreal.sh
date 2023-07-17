#!/bin/bash

# trajectory visualization of Unreal-RS datasets
CUDA_VISIBLE_DEVICES=0 python traj_process_Unreal.py --config ./configs/Unreal-RS/shake1_CubicSpline_2.txt

# Unreal-RS datasets high framerate video
CUDA_VISIBLE_DEVICES=0 nohup python test_video_Unreal_4.py --config ./configs/Unreal-RS/test_syn_4_video_shake1.txt >> nohup/Unreal-RS/test_syn_4_video_shake1.log 2>&1 &

# Unreal-RS datasets novel view synthesis
CUDA_VISIBLE_DEVICES=0 nohup python test_novel_view_synthesis.py --config ./configs/Unreal_novel_view/Blue_Room.txt >> nohup/Unreal_novel_view/Blue_Room.log 2>&1 &

# Unreal-RS datasets training
CUDA_VISIBLE_DEVICES=0 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Unreal-RS/shake1_CubicSpline_2.txt >> nohup/Unreal-RS/shake1_CubicSpline_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train_Unreal_CubicSpline_4.py --config ./configs/Unreal-RS/shake1_CubicSpline_4.txt >> nohup/Unreal-RS/shake1_CubicSpline_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Living-Room-1000Hz.txt >> nohup/Synthetic-Datasets/Living-Room-1000Hz.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Living-Room-1000Hz.txt >> nohup/Synthetic-Datasets/Living-Room-1000Hz-rgb_loss.log 2>&1 &

# event SLAM
CUDA_VISIBLE_DEVICES=1 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-groundtruth/Living-Room-1000Hz-no-rgb_loss-fixing-window.txt >> nohup/Synthetic-Datasets/Living-Room-1000Hz-no-rgb_loss-fixing-window.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-groundtruth/Living-Room-1000Hz-no-rgb_loss-sliding-window.txt >> nohup/Synthetic-Datasets/Living-Room-1000Hz-no-rgb_loss-sliding-window.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-groundtruth/Living-Room-1000Hz-no-rgb_loss-sliding-window-optimize-pose.txt >> nohup/Synthetic-Datasets/Living-Room-1000Hz-no-rgb_loss-sliding-window-optimize-pose.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-groundtruth/Living-Room-1000Hz-no-rgb_loss-fixing-window-optimize-pose.txt >> nohup/Synthetic-Datasets/Living-Room-1000Hz-no-rgb_loss-fixing-window-optimize-pose.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-mid/fixing-window-optimize-pose-50ms.txt >> nohup/Synthetic-Datasets-initialize-with-mid/fixing-window-optimize-pose-50ms.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-mid/fixing-window-optimize-pose-5ms.txt >> nohup/Synthetic-Datasets-initialize-with-mid/fixing-window-optimize-pose-5ms.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_Unreal_CubicSpline_2_chunk.py --config ./configs/Synthetic-Datasets-initialize-with-mid/50ms-chunk.txt >> nohup/Synthetic-Datasets-initialize-with-mid/50ms-chunk.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-mid/50ms-chunk-whole-trajectory.txt >> nohup/Synthetic-Datasets-initialize-with-mid/50ms-chunk-whole-trajectory.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-mid/50ms-chunk-whole-trajectory-barf.txt >> nohup/Synthetic-Datasets-initialize-with-mid/50ms-chunk-whole-trajectory-barf.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train_Unreal_CubicSpline_2.py --config ./configs/Synthetic-Datasets-initialize-with-mid/50ms-chunk-whole-trajectory-no-barf.txt >> nohup/Synthetic-Datasets-initialize-with-mid/50ms-chunk-whole-trajectory-no-barf.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python test_Unreal_CubicSpline_2.py --config ./configs/Test/50ms-chunk-whole-trajectory.txt >> nohup/Test/50ms-chunk-whole-trajectory.log 2>&1 &