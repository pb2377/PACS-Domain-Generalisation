import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    # Data and model
    parser.add_argument('--target', default=None, help='PACS DG target domain.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size.')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--net', default='resnet18', choices=['resnet18', 'resnet50', 'alexnet'],
                        help='Model architecture')
    parser.add_argument('--jitter', default=0.4, type=float, help="Color jitter amount")
    parser.add_argument('--reps', default=5, type=int, help='Number of repeats on each test.')
    parser.add_argument('--domainnet', default=False,
                        action='store_true', help='Flag to train on domainnet instead of PACS.')

    # optimizer parameters
    parser.add_argument('--lr', '--learning_rate', type=float, default=.001, help="Optimizer Learning rate")
    parser.add_argument('--epochs', '-e', type=int, default=30, help="Number of epochs")
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    # Adaptation args
    parser.add_argument('--adapt', default=False, action='store_true',
                        help='Domain adaptation rather than generalisation.')
    return parser.parse_args()


def get_optimizer(args, params):
    lr_scheduler = None
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.epochs > 5:
        step_size = int(args.epochs * .6)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    return optimizer, lr_scheduler


def get_setup(args):
    if args.domainnet:
        domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        raise NotImplementedError
    else:
        domain_list = ['photo', 'art_painting', 'cartoon', 'sketch']

    setups = {tar: [sc for sc in domain_list if sc != tar] for tar in domain_list}
    if args.target is not None:
        assert args.target in domain_list
        setups = {k: d for k, d in setups.items() if k == args.target}

    return setups
