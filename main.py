import os
import time
import argparse

import torch.nn as nn

from trainer import Trainer
from networks import build_model
from datasets import get_dataloaders
from utils import get_optimizer


def main(args, rep=0):
    # Get model
    model = build_model(args)

    # get dataset
    train_loader, test_loaders = get_dataloaders(args)

    # build trainer
    logdir = os.path.join('save', args.net,
                          'domainnet' if args.domainnet else 'pacs',
                          '{}-{}'.format(args.target, rep_id))
    trainer = Trainer(logdir)

    optimizer, lr_scheduler = get_optimizer(args, model.parameters())
    criterion = nn.CrossEntropyLoss()

    trainer.train(model, criterion, optimizer, lr_scheduler, train_loader, test_loaders, epochs=args.epochs)
    trainer.print_report(rep)


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


if __name__ == '__main__':
    args = get_args()

    if args.domainnet:
        domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        raise NotImplementedError
    else:
        domain_list = ['photo', 'art_painting', 'cartoon', 'sketch']

    setups = {tar: [sc for sc in domain_list if sc != tar] for tar in domain_list}
    if args.target is not None:
        assert args.target in domain_list
        setups = {k: d for k, d in setups.items() if k == args.target}

    for target, source in setups.items():
        assert target not in source
        args.sources = source
        args.target = target
        print("\nTarget domain: {} \tSource domains: {}".format(args.target, args.sources))
        avg_rep_time = 0.
        for rep_id in range(args.reps):
            t0 = time.time()
            main(args, rep_id)
            avg_rep_time += time.time() - t0

        avg_rep_time /= 60
        print('Total Time {:.1f} minutes \t '
              'Average rep time {:.1f} minutes'.format(avg_rep_time, avg_rep_time / args.reps))
