#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/40.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/41.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/42.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/43.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/44.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/45.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/46.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/47.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/48.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/49.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/50.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/51.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/mic/52.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/mic/53.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/mic/54.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait


