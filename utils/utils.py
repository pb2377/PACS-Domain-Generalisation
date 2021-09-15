import torch


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
