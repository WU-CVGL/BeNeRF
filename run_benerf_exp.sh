#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/materials/8.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/materials/9.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/materials/10.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/11.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/materials/12.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/materials/13.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/materials/14.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/materials/15.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/materials/16.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/materials/17.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/materials/18.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/19.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/materials/20.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/materials/21.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/materials/22.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/materials/23.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/materials/24.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/materials/25.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/materials/26.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/27.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/materials/28.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/materials/29.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/materials/30.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/materials/31.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/materials/32.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/materials/33.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/materials/34.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/35.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/materials/36.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/materials/37.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/materials/38.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/materials/39.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/materials/40.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/materials/41.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/materials/42.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/43.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/materials/44.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/materials/45.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/materials/46.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/materials/47.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait


