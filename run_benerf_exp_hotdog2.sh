#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/hotdog/51.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/hotdog/52.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/hotdog/53.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/hotdog/54.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/hotdog/55.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/hotdog/56.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/hotdog/57.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/hotdog/58.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/hotdog/59.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/hotdog/61.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/hotdog/62.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/hotdog/63.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/hotdog/64.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/hotdog/65.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/hotdog/66.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/hotdog/67.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/hotdog/68.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/hotdog/69.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/hotdog/71.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/hotdog/72.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/hotdog/73.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/hotdog/74.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/hotdog/75.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/hotdog/76.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/hotdog/77.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/hotdog/78.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/hotdog/79.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/hotdog/81.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/hotdog/82.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/hotdog/83.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/hotdog/84.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/hotdog/85.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/hotdog/86.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/hotdog/87.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/hotdog/88.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/hotdog/89.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/hotdog/91.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/hotdog/92.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/hotdog/93.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/hotdog/94.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/hotdog/95.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/hotdog/96.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait






