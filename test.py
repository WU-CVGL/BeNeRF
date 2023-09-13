import os
import random

import numpy as np
import torch

from config import config_parser


def test(args):
    pass


if __name__ == '__main__':
    # load config
    print("Loading config")
    parser = config_parser()
    args = parser.parse_args()

    # setup seed (for exp)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['PYTHONHASHSEED'] = str(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.random.manual_seed(0)
    if not args.debug:
        # performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup device
    print(f"Use device: {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # train
    print("Start training...")
    test(args)
