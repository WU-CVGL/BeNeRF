import os
import h5py
import torch
import hdf5plugin
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import img_utils

def load_img_data(datadir, datasource = None, gray = False):

    # Load images
    imgdir = os.path.join(datadir, "images")
    if datasource == "BeNeRF_Blender" or datasource == "BeNeRF_Unreal" or datasource == "E2NeRF_Synthetic":
        testdir = os.path.join(datadir, "images" + "_test")

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.lower().endswith(("jpg", "png"))
    ]

    imgtests_file = []
    if datasource == "BeNeRF_Blender" or datasource == "BeNeRF_Unreal" or datasource == "E2NeRF_Synthetic":
        imgtests_file = [
            os.path.join(testdir, f)
            for f in sorted(os.listdir(testdir))
            if f.lower().endswith(("jpg", "png"))
        ]

    # Using iterative approach to read image into a NumPy array helps to reduce memory cost.
    # Converting the entire image list into a NumPy array may result in a memory overflow.
    if gray == True:
        h, w = img_utils.load_image(imgfiles[0], gray).shape
        imgs = np.empty((len(imgfiles), h, w), dtype = np.float64)
        for i in tqdm(range(len(imgfiles))):
            imgs[i, :, :] = img_utils.load_image(imgfiles[i], gray)
    elif gray == False:
        h, w, _ = img_utils.load_image(imgfiles[0], gray).shape
        imgs = np.empty((len(imgfiles), h, w, 3), dtype = np.float64)
        for i in tqdm(range(len(imgfiles))):
            imgs[i, :, :, :] = img_utils.load_image(imgfiles[i], gray)
    
    imgtests = []
    if datasource == "BeNeRF_Blender" or datasource == "BeNeRF_Unreal" or datasource == "E2NeRF_Synthetic":
        if gray == True:
            h, w = img_utils.load_image(imgtests_file[0], gray).shape
            imgtests = np.empty((len(imgtests_file), h, w), dtype = np.float64)
            for i in tqdm(range(len(imgtests_file))):
                imgtests[i, :, :] = img_utils.load_image(imgtests_file[i], gray)
        elif gray == False:
            h, w, _ = img_utils.load_image(imgtests_file[0], gray).shape
            imgtests = np.empty((len(imgtests_file), h, w, 3), dtype = np.float64)
            for i in tqdm(range(len(imgtests_file))):
                imgtests[i, :, :, :] = img_utils.load_image(imgtests_file[i], gray)

    return imgs, imgtests

def load_camera_pose(basedir, H, W, cubic):
    sh = H, W
    # load poses
    if cubic:
        poses_arr = np.load(os.path.join(basedir, "poses_bounds_cubic.npy"))
        ev_poses_arr = np.load(os.path.join(basedir, "poses_bounds_cubic_events.npy"))
    else:
        poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
        ev_poses_arr = np.load(os.path.join(basedir, "poses_bounds_events.npy"))

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3x5xN
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
    )  # 列的转换    -y x z : x y z
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)

    ev_poses = ev_poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    ev_poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    ev_poses = np.concatenate(
        [ev_poses[:, 1:2, :], -ev_poses[:, 0:1, :], ev_poses[:, 2:, :]], 1
    )
    ev_poses = np.moveaxis(ev_poses, -1, 0).astype(np.float32)

    return poses, ev_poses

def load_camera_trans(basedir):
    # load trans
    trans_arr = np.load(os.path.join(basedir, "trans.npy"))
    return trans_arr

def load_timestamps(basedir, args):

    # BeNeRF_Synthetic 
    if args.dataset in ["BeNeRF_Blender", "BeNeRF_Unreal"]:
        time_ts_path = os.path.join(basedir, "poses_ts.txt")
        times_ts = np.loadtxt(time_ts_path)
        times_start = times_ts[:-1]
        times_end = times_ts[1:]
    # TUM-VIE
    elif args.dataset == "TUM_VIE":
        timestamps_path = os.path.join(basedir, "image_timestamps.txt")
        exposures_path = os.path.join(basedir, "image_exposures.txt")
        timestamps = np.loadtxt(timestamps_path)
        exposures = np.loadtxt(exposures_path)
        times_start = timestamps[:] - 0.5 * exposures[:]
        times_end = timestamps[:] + 0.5 * exposures[:]
    # E2NeRF_Real
    elif args.dataset == "E2NeRF_Real":
        time_start_path = os.path.join(basedir, "exposure_start_ts.txt")
        time_end_path = os.path.join(basedir, "exposure_end_ts.txt")
        times_start = np.loadtxt(time_start_path)
        times_end = np.loadtxt(time_end_path)
    # E2NeRF_Synthetic
    elif args.dataset == "E2NeRF_Synthetic":
        eventdir = os.path.join(basedir, "events")
        eventdir_idx = eventdir + "/r_{}".format(args.index * 2)
        events_txt = np.loadtxt(os.path.join(eventdir_idx, "v2e-dvs-events.txt"))
        st, _, _ ,_ = events_txt[0]
        ed, _, _ ,_ = events_txt[len(events_txt) - 1]
        times_start = int(st * 1e19)
        times_end = int(ed * 1e19)
    else:
        print("Cannot load timestamps for images")
        assert False

    if args.dataset == "E2NeRF_Synthetic":
        # record exposure time for rgb camera
        img_ts_start = times_start
        img_ts_end = times_end
        # usually,select more events will be better
        evt_ts_start = times_start - args.event_shift_start * 1e3 
        evt_ts_end = times_end + args.event_shift_end * 1e3
    else:
        # record exposure time for rgb camera    
        img_ts_start = times_start[args.index]
        img_ts_end = times_end[args.index]
        # usually,select more events will be better
        evt_ts_start = times_start[args.index] - args.event_shift_start * 1e3 
        evt_ts_end = times_end[args.index] + args.event_shift_end * 1e3

    return img_ts_start, img_ts_end, evt_ts_start, evt_ts_end

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds

def load_data(
    datadir, args, load_pose = False, load_trans = False, cubic = False, datasource = None
):
    datadir = os.path.expanduser(datadir)
    gray = args.channels == 1

    # load imges
    # [num, height, width, channel]
    print("Loading images...")
    img = []            # input
    imgtest = []        # groundtruth
    imgs, imgtests = load_img_data(datadir, datasource, gray = gray)
    if gray:
        imgs = np.expand_dims(imgs, -1)
    # select one image: [1, height, width, channel]
    img = np.expand_dims(imgs[args.index], 0)

    if datasource in ["BeNeRF_Blender", "BeNeRF_Unreal", "E2NeRF_Synthetic"]:
        if gray:
            imgtests = np.expand_dims(imgtests, -1)
        # select one image
        imgtest = np.expand_dims(imgtests[args.index], 0)
    print("Load images successfully!!")

    # load start and end timestamps of exposure time
    print("Loading timestamps...")
    img_ts_start, img_ts_end, evt_ts_start, evt_ts_end = load_timestamps(datadir, args)
    print("Load timestamps successfully!!")

    # load events
    print("Loading events...")
    eventdir = os.path.join(datadir, "events")
    # BeNeRF Synthetic
    if datasource in ["BeNeRF_Blender", "BeNeRF_Unreal"]:
        events = np.load(os.path.join(eventdir, "events.npy"))
        events = np.array(
            [ event for event in events if evt_ts_start <= event[2] <= evt_ts_end]
        )
    # E2NeRF Real
    elif datasource == "E2NeRF_Real":
        events_tensor = torch.load(os.path.join(eventdir, "events.pt"))
        events_numpy = events_tensor.numpy()
        events =  np.array(
            [event for event in tqdm(events_numpy) if evt_ts_start <= event[2] <= evt_ts_end]
        )
    # E2NeRF_Synthetic
    elif datasource == "E2NeRF_Synthetic":
        eventdir_idx = eventdir + "/r_{}".format(args.index * 2)
        events_txt = np.loadtxt(os.path.join(eventdir_idx, "v2e-dvs-events.txt"))
        events_list = []
        for row in tqdm(events_txt):
            t, x, y, p = row
            t = t * 1e19
            p = 2 * p - 1
            events_list.append(np.array([x, y, t, p], dtype = np.int64))
        events = np.array(events_list)
    # TUM-VIE
    elif datasource == "TUM_VIE":
        # import h5 file
        h5file = h5py.File(os.path.join(eventdir, "events.h5"))
        # h5group contains h5dataset: [x y t p]
        h5group = h5file["events"]

        # select events corresponding to idx
        h5dataset_ts = h5group["t"]

        # iteratively import timestamps of event data in chunks
        selected_indices = np.array([])
        chunk_size = 500000
        for chunk_idx in tqdm(range(0, len(h5dataset_ts), chunk_size)):
            chunk_indices = np.where(
                (h5dataset_ts[chunk_idx : chunk_idx + chunk_size] >= evt_ts_start) 
                & (h5dataset_ts[chunk_idx : chunk_idx + chunk_size] <= evt_ts_end)
            )
            chunk_indices = chunk_indices[0]
            chunk_indices[:] = chunk_indices[:] + chunk_idx
            selected_indices = np.concatenate((selected_indices, chunk_indices)).astype(np.uint64)
        selected_indices_start = np.array(selected_indices[0], dtype = np.uint64)
        selected_indices_end = np.array(selected_indices[len(selected_indices) - 1] + 1, dtype = np.uint64)
        # creat events array
        events = np.zeros(len(selected_indices))
        h5group_order = ["x", "y", "t", "p"]
        for i in tqdm(range(len(h5group_order))):
            h5dataset_name = h5group_order[i]
            h5dataset = h5group[h5dataset_name][
                selected_indices_start : selected_indices_end
            ]
            events = np.vstack((events, h5dataset))
        events = np.delete(events, 0, axis = 0)
        events = np.transpose(events)

    # sorted according to time
    events = events[events[:, 2].argsort()]
    # create dictionary
    events = {
        "x": events[:, 0].astype(int),
        "y": events[:, 1].astype(int),
        # norm ts(0~1)
        "ts": (events[:, 2] - evt_ts_start) / (evt_ts_end - evt_ts_start),
        "pol": events[:, 3],
    }
    print("Load events successfully!!")
    # process poses
    poses, ev_poses, trans, poses_ts = None, None, None, None
    if load_pose:
        poses, ev_poses = load_camera_pose(datadir, imgs.shape[0], imgs.shape[1], cubic)
        # recenter for rgb
        poses_num = 4 if cubic else 2
        poses_all = np.concatenate(
            (poses[args.index : args.index + 2], ev_poses[args.index : args.index + 2]), axis=0
        )
        poses_all = recenter_poses(poses_all)
        poses = poses_all[0:poses_num]

        # recenter for event
        ev_poses = poses_all[poses_num : 2 * poses_num]
    elif load_trans:
        trans_arr = load_camera_trans(datadir)

        trans = trans_arr.astype(np.float32)

    # normlize exposure time of image accroding to eventstream time
    img_ts_start = (img_ts_start - evt_ts_start) / (evt_ts_end - evt_ts_start)
    img_ts_end = (img_ts_end - evt_ts_start) / (evt_ts_end - evt_ts_start)
    rgb_exp_time = np.array([img_ts_start, img_ts_end])

    return events, img, imgtest, rgb_exp_time, poses_ts, poses, ev_poses, trans

def regenerate_pose(
    poses, bds, recenter=True, bd_factor=0.75, spherify=False, path_zflat=False
):
    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)
        # print('recentered', c2w.shape)
        # print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = 0.8
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)

    render_poses = torch.Tensor(render_poses)

    return render_poses
