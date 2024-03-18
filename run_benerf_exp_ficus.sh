#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/ficus/87.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/ficus/9.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/ficus/0.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/ficus/1.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/ficus/2.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

# 等待所有后台任务完成
wait







