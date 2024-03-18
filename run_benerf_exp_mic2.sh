#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/20.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/21.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/22.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/mic/23.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

# 等待所有后台任务完成
wait

# 继续执行下一段的8个命令
# 如果有更多的命令，可以继续添加到commands数组中
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/24.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/25.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/26.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/mic/27.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/28.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/29.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/30.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/mic/31.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/32.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/33.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/34.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/mic/35.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/36.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/37.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/38.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/mic/39.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait


