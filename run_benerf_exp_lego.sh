#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/88.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/89.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/91.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/92.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/93.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/94.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/95.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/96.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/97.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/98.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/99.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/45.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait







