import torch


def get_optimizer(args, params):
    lr_scheduler = None
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.epochs > 5:
        step_size = int(args.epochs * .6)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    return optimizer, lr_scheduler
