#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/ficus/3.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/ficus/4.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/ficus/5.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

# 等待所有后台任务完成
wait

