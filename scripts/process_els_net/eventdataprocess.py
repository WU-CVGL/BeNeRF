import scipy.io as scio
import numpy as np
import os


def find_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def load_timestamps(basedir):
    poses_ts_path = os.path.join(basedir, 'poses_ts.txt')
    poses_start_path = os.path.join(basedir, "poses_start_ts.txt")
    poses_end_path = os.path.join(basedir, "poses_end_ts.txt")
    if os.path.exists(poses_ts_path):
        poses_ts = np.loadtxt(poses_ts_path)
        poses_start = poses_ts[:-1]
        poses_end = poses_ts[1:]
    elif os.path.exists(poses_start_path) and os.path.exists(poses_end_path):
        poses_start = np.loadtxt(poses_start_path)
        poses_end = np.loadtxt(poses_end_path)
    else:
        print("Cannot load timestamps for images")
        assert False

    return poses_start, poses_end


def process_data(datadir):
    # process events and timestamps
    ts_start, ts_end = load_timestamps(datadir)
    NUM = len(ts_end)
    eventdir = os.path.join(datadir, "events")

    event_list = []
    if os.path.exists(os.path.join(eventdir, "events.npy")):
        for i in range(NUM):
            poses_ts = np.array((ts_start[i], ts_end[i]))
            events = np.load(os.path.join(eventdir, "events.npy"))
            delta = (poses_ts[1] - poses_ts[0]) * 0
            poses_ts = np.array([poses_ts[0] - delta, poses_ts[1] + delta])

            events = np.array([event for event in events if poses_ts[0] <= event[2] <= poses_ts[1]])

            event_list.append(events)

    return event_list


def generate_mat(base_path):
    scenes = os.listdir(base_path)

    for scene in scenes:
        scene_path = os.path.join(base_path, scene)
        events = process_data(scene_path)
        # events = np.zeros([20, 100, 4])
        for i in range(len(events)):

            event_data = events[i]
            section_event_timestamp = np.array(np.expand_dims(event_data[:, 2], 0))

            event_mat = {
                         'section_event_timestamp': section_event_timestamp,
                         'section_event_x': np.array(np.expand_dims(event_data[:, 0], 0).astype('int64')),
                         'section_event_y': np.array(np.expand_dims(event_data[:, 1], 0).astype('int64')),
                         'section_event_polarity': np.array(np.expand_dims((event_data[:, 3] + 1) // 2, 0).astype('int64')),
                         'start_timestamp': np.array(np.expand_dims(section_event_timestamp[:, 0], 0)),
                         'end_timestamp': np.array(np.expand_dims(section_event_timestamp[:, -1], 0))
                         }
            event_mat_path= os.path.join(scene_path, "mat")
            os.makedirs(event_mat_path, exist_ok=True)
            save_path = os.path.join(event_mat_path, 'event{:06d}.mat'.format(i))

            scio.savemat(save_path, event_mat)

        print('event mat has been generated')


if __name__ == "__main__":
    folder_path = r'D:\learn\paper\2023cvpr\script\wpdata\test_data'
    generate_mat(folder_path)
