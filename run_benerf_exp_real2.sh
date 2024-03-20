#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/letter/15.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/letter/16.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/letter/17.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/letter/18.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/letter/19.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/letter/20.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/letter/21.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/letter/22.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/real/letter/23.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/letter/24.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/letter/25.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/letter/26.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/letter/27.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/letter/28.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/letter/29.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/plant/8.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/plant/9.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/plant/10.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/plant/11.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/plant/12.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/plant/13.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/plant/14.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/plant/15.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/plant/16.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/plant/17.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/plant/18.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/plant/19.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/plant/20.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/plant/21.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/plant/22.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/plant/23.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/plant/24.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/real/plant/25.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/real/plant/26.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/real/plant/27.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/real/plant/28.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/real/plant/29.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/real/toys/8.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/real/toys/9.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/real/toys/10.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

# 等待所有后台任务完成
wait




