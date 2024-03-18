#!/bin/bash

# 待执行的命令列表
commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/18.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/19.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/21.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/22.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/lego/48.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/lego/49.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/lego/51.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/lego/52.txt' &"
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
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/53.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/54.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/55.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/56.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/lego/23.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/lego/24.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/lego/25.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/lego/26.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/27.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/28.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/29.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/31.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/lego/32.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/lego/33.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/lego/34.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/lego/35.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/36.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/37.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/38.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/57.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/lego/58.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/lego/59.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/lego/61.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/lego/62.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/63.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/64.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/65.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/66.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/lego/67.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/lego/68.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/lego/69.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/lego/71.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/72.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/73.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/74.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/75.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/lego/76.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/lego/77.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/lego/78.txt' &"
    "python train.py --device 7 --config './configs/e2nerf/synthetic/lego/79.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait

commands=(
    "python train.py --device 0 --config './configs/e2nerf/synthetic/lego/81.txt' &"
    "python train.py --device 1 --config './configs/e2nerf/synthetic/lego/82.txt' &"
    "python train.py --device 2 --config './configs/e2nerf/synthetic/lego/83.txt' &"
    "python train.py --device 3 --config './configs/e2nerf/synthetic/lego/84.txt' &"
    "python train.py --device 4 --config './configs/e2nerf/synthetic/lego/85.txt' &"
    "python train.py --device 5 --config './configs/e2nerf/synthetic/lego/86.txt' &"
    "python train.py --device 6 --config './configs/e2nerf/synthetic/lego/87.txt' &"
)

# 并行执行8个命令
for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait



