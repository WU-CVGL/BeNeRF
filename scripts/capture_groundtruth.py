import os
import numpy as np

def generate_rs(imgdir, basedir):
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    num = len(imgfiles)
    select_pose_dir = os.path.join(basedir, 'groundtruth.txt')
    load_gt = np.loadtxt(os.path.join(basedir, 'spline_groundtruth.txt'))

    # syn with images
    start_timestamp = float(os.listdir(imgdir)[0][:-4])
    start_pose = np.where(load_gt[..., 0] == start_timestamp)
    np.savetxt(select_pose_dir, load_gt[start_pose[0][0]: int(start_pose[0] + num), ...], fmt='%.8f')

if __name__=='__main__':
    basedir = r'D:\wp-gen\whiteroom'
    imgdir = os.path.join(basedir, 'camera/temp')
    generate_rs(imgdir, basedir)

