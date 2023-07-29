import os.path

import esim_py

contrast_threshold_pos = 0.1
contrast_threshold_neg = 0.1
refractory_period = 1e-9
log_eps = 1e-3
use_log = False

imgdir = os.path.expanduser("./data/event-datasets-origin/LivingRoom/camera/temp")
outdir = os.path.expanduser("./data/livingroom_output")

if __name__ == '__main__':
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.lower().endswith(('jpg', 'png', 'jpeg'))]

    list_of_timestamps = [float(os.path.splitext(os.path.basename(f))[0]) for f in imgfiles]

    # constructor
    esim = esim_py.EventSimulator(
        contrast_threshold_pos,  # contrast thesholds for positive
        contrast_threshold_neg,  # and negative events
        refractory_period,  # minimum waiting period (in sec) before a pixel can trigger a new event
        log_eps,  # epsilon that is used to numerical stability within the logarithm
        use_log,  # wether or not to use log intensity
    )

    # setter, useful within a training loop
    esim.setParameters(contrast_threshold_pos, contrast_threshold_neg, refractory_period, log_eps, use_log)

    # generate events from list of images and timestamps
    events_list_of_images = esim.generateFromStampedImageSequence(
        imgfiles,  # list of absolute paths to images
        list_of_timestamps  # list of timestamps in ascending order
    )

    print(events_list_of_images)
