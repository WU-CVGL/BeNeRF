import numpy as np
import os

if __name__ == '__main__':
    basedir = 'C:/Users/17120/Desktop/Event Camera/Datasets/Synthetic datasets'
    dataset_name = sorted(os.listdir(basedir))
    dataset_dir = [os.path.join(basedir, folder) for folder in dataset_name if not folder.endswith('zip')]

    for data_dir in dataset_dir:
        events_dir = os.path.join(data_dir, 'events')
        event_npy_dir = [os.path.join(events_dir, file) for file in sorted(os.listdir(events_dir)) if
                         file.endswith('npy')]
        for f_event_dir in event_npy_dir:
            f_event = np.load(f_event_dir)  # event.npy format [x ,y , t_us, polarity, 0]
            print(0)
