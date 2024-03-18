#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/6.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/7.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/20.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/mic/21.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/mic/23.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/mic/40.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/mic/41.txt' &"

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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/42.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/55.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/56.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/mic/57.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/mic/60.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/mic/74.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/mic/75.txt' &"

)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait





