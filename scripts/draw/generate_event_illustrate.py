import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.random.seed(90)
    n = int(3e3)
    ts = np.random.rand(n) * 10
    x = np.random.randint(0, high=255, size=n)
    y = np.random.randint(0, high=255, size=n)
    p = np.random.randint(0, high=2, size=n) * 2 - 1
    fig = plt.figure(dpi=800)
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b' if p_ == 1 else 'r' for p_ in p]

    ax.scatter3D(ts, x, y, c=colors, s=1)
    ax.view_init(elev=10, azim=270)  # 更改视角的仰角和方位角
    ax.set_box_aspect([4, 1, 1])
    # ax.set_axis_off()
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 设置x轴线为透明
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 设置z轴线为透明
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 设置z轴线为透明

    plt.show()
