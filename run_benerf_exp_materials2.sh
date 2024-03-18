#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/materials/0.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/materials/1.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/materials/2.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/3.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/materials/4.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/materials/5.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/materials/6.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/materials/7.txt' &"
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

wait

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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/materials/48.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/materials/49.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/materials/50.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/materials/51.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/materials/52.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/materials/53.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/materials/54.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/materials/55.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait



