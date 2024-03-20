#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/toys/22.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/toys/23.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/toys/24.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/toys/25.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/real/toys/26.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/toys/27.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/toys/28.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/toys/29.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait






