def argument_print(args, checkpoint_name):

    print('-----------------------------------')
    print('(1) Dataset: ', args.dataset)
    print('(2) network: ', args.network)
    print('(3) attack: ', args.attack)
    print('(4) MaxNorm: ', args.eps)
    print('(5) epoch: ', args.epoch)
    print('(6) checkpoint: ', checkpoint_name)
    print('-----------------------------------')


def argument_testprint(args, checkpoint_name):

    print('-----------------------------------')
    print('(1) Dataset: ', args.dataset)
    print('(2) network: ', args.network)
    print('(3) MaxNorm: ', args.eps)
    print('(4) steps: ', args.steps)
    print('(5) checkpoint: ', checkpoint_name)
    print('-----------------------------------')
