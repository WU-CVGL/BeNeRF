#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/camera/19.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/camera/20.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/camera/21.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/camera/22.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/camera/23.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/camera/24.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/camera/25.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/camera/26.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/real/camera/27.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/camera/28.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/camera/29.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/lego/8.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/lego/9.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/lego/10.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/lego/11.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/lego/12.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/lego/13.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/lego/14.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/lego/15.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/lego/16.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/lego/17.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/lego/18.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/lego/19.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/lego/20.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/lego/21.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/lego/22.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/lego/23.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/lego/24.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/lego/25.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/lego/26.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/lego/27.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/lego/28.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/lego/29.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/letter/8.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/letter/9.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/letter/10.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/letter/11.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/letter/12.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/letter/13.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/letter/14.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

# 等待所有后台任务完成
wait




