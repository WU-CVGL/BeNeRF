import os

import torch

from config import config_parser
from model import nerf_cubic_optimpose, nerf_cubic_optimtrans, nerf_cubic_optimposeset, nerf_cubic_rigidtrans, \
    nerf_linear_optimpose, nerf_linear_optimtrans, nerf_linear_optimposeset, test_model


def get_event_traj(graph, args, n):
    """
    :param graph: Input model
    :param args: config args
    :param n: number of pose to generate
    :return: event poses
    """
    events_ts = torch.arange(1e-6, 1, n)
    return graph.get_pose(args, events_ts)


def get_rgb_traj(graph, args, n):
    """
    :param graph: Input model
    :param args: config args
    :param n: number of pose to generate
    :return: rgb poses
    """

    return graph.get_pose_rgb(args, n)


def main(args):
    # choose model
    if args.model == "cubic_optimpose":
        model = nerf_cubic_optimpose.Model(args)
    elif args.model == "cubic_optimtrans":
        model = nerf_cubic_optimtrans.Model(args)
    elif args.model == "cubic_optimposeset":
        model = nerf_cubic_optimposeset.Model(args)
    elif args.model == "cubic_rigidtrans":
        model = nerf_cubic_rigidtrans.Model(args)
    elif args.model == "linear_optimpose":
        model = nerf_linear_optimpose.Model(args)
    elif args.model == "linear_optimtrans":
        model = nerf_linear_optimtrans.Model(args)
    elif args.model == "linear_optimposeset":
        model = nerf_linear_optimposeset.Model(args)
    elif args.model == "test":
        model = test_model.Model(args)
    else:
        print("Unknown model type")
        return

    print(f"Use model type {args.model}")

    logdir = os.path.join(os.path.expanduser(args.logdir), args.expname)

    if args.load_weights:
        graph = model.build_network(args)
        path = os.path.join(logdir, '{:06d}.tar'.format(args.weight_iter))
        graph_ckpt = torch.load(path)

        graph.load_state_dict(graph_ckpt['graph'])
        print('Model Load Done!')
    else:
        raise AssertionError("NOT LOAD WEIGHTS")

    # get rgb pose x 1000
    rgb_poses = get_rgb_traj(graph, args, 1000)
    # get event pose x1000
    event_poses = get_event_traj(graph, args, 1000)


if __name__ == '__main__':
    # load config
    print("Loading config")
    parser = config_parser()
    args = parser.parse_args()

    # trajectory
    print("Start trajectory generation")
    main(args=args)
