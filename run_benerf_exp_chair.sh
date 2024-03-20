#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/17.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/18.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/19.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/21.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/22.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/23.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/24.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/25.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/26.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/27.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/28.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/29.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/31.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/32.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/33.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/34.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/35.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/36.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/37.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/38.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/39.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/41.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/42.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/43.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/44.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/45.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/46.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/47.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/48.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/49.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/51.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/52.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/chair/53.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/chair/54.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/chair/55.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/chair/56.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/chair/57.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/chair/58.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/chair/59.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/chair/61.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

# 等待所有后台任务完成
wait




