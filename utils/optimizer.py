import torch.optim as optim


def optimizer(args, parameters):
    if args.optimizer.lower() == "adam":
        return optim.Adam([p for p in parameters if p.requires_grad], lr=args.learning_rate_f,
                          weight_decay=args.weight_decay_f)
    elif args.optimizer.lower() == "sgd":
        return optim.SGD([p for p in parameters if p.requires_grad], lr=args.learning_rate_f,
                         weight_decay=args.weight_decay_f)
    else:
        assert False, "no application for the optimizer"
