#!/bin/bash

# 继续执行下一段的8个命令
# 如果有更多的命令，可以继续添加到commands数组中
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/62.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/63.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/64.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/65.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/66.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/67.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/68.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/69.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/71.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/72.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/73.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/74.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/75.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/76.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/77.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/78.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/79.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/81.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/82.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/83.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/84.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/85.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/86.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/87.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/88.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/89.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/91.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/92.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/93.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/94.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/95.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/96.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/97.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/98.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/hotdog/97.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/hotdog/98.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/hotdog/99.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/99.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/mic/98.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/mic/99.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait




