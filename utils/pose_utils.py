import numpy
import torch
import os

@torch.no_grad()
def save_poses_as_kitti_format(iter_step, logdir, poses_matrices):
    poses_dir = os.path.join(logdir, "poses_test")
    os.makedirs(poses_dir,exist_ok=True)
    poses_file = os.path.join(poses_dir, "poses_test_{:06d}.txt".format(iter_step))
    with open(poses_file, 'w') as file:
        for pose_matrix in poses_matrices:
            for i, row in enumerate(pose_matrix):
                row = row.cpu().numpy()
                row = row.tolist()
                row_str = ' '.join(map(str, row))
                if i != 2:
                    file.write(row_str + ' ')
                else:
                    file.write(row_str)
            file.write('\n')


