import os
import time

import torch.nn as nn

from datasets import get_dataloaders
from networks import build_model
from trainer import Trainer
from utils import get_optimizer, get_setup, get_args


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

    # train, log and report DG performance
    trainer.train(model, criterion, optimizer, lr_scheduler, train_loader, test_loaders, epochs=args.epochs)
    trainer.print_report(rep)


if __name__ == '__main__':
    args = get_args()
    domain_setups = get_setup(args)
    for target, source in domain_setups.items():
        assert target not in source
        args.sources = source
        args.target = target
        print("\nTarget domain: {}\tSource domains: {}".format(args.target, args.sources))
        avg_rep_time = 0.

        # Repeat the initialise and train, as artistic DG often varies a lot run to run.
        for rep_id in range(args.reps):
            t0 = time.time()

            # Main fn
            main(args, rep_id)

            avg_rep_time += time.time() - t0

        # report how long each training rep took.
        avg_rep_time /= 60
        print('\tTotal Time {:.1f} minutes \t '
              '({:.1f} minutes avg)'.format(avg_rep_time, avg_rep_time / args.reps))
