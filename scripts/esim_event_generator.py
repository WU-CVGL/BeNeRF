import esim_torch
import os
import cv2
import numpy as np
import torch

contrast_threshold_neg = 0.1
contrast_threshold_pos = 0.1
refractory_period_ns = 1e3
imgdir = os.path.expanduser("D:\\dataset\\LivingRoom\\camera\\test")
eventdir = os.path.expanduser("D:\\dataset\\LivingRoom\\camera\\teste")


# # event generation
# events = esim.forward(
#     log_images,        # torch tensor with type float32, shape T x H x W
#     timestamps_ns  # torch tensor with type int64,   shape T
# )
#
# # Reset the internal state of the simulator
# events.reset()

# events can also be generated in a for loop
# to keep memory requirements low
def generate(log_images):
    # constructor
    esim = esim_torch.ESIM(
        contrast_threshold_neg,  # contrast threshold for negative events
        contrast_threshold_pos,  # contrast threshold for positive events
        refractory_period_ns  # refractory period in nanoseconds
    )

    idx = 0
    for log_image in log_images:
        img = torch.tensor(cv2.imread(log_image, cv2.IMREAD_GRAYSCALE) / 255., dtype=torch.float32)
        # time in s
        file_name = os.path.splitext(os.path.basename(log_image))[0]
        ts_ns = torch.tensor(round((float(file_name)) * 1e9), dtype=torch.int64)

        sub_events = esim.forward(img, ts_ns)

        # for the first image, no events are generated, so this needs to be skipped
        if sub_events is None:
            continue

        # do something with the events
        x = sub_events["x"].cpu().numpy()
        y = sub_events["y"].cpu().numpy()
        t = sub_events["t"].cpu().numpy() * 1e-9
        p = sub_events["p"].cpu().numpy()
        np.save(os.path.join(eventdir, "{:06d}.npy".format(idx)), np.array((x, y, t, p)))

        idx += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.random.manual_seed(0)

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    generate(imgfiles)
