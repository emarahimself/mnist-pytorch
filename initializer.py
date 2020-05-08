import torch


def random_init(n1, n2):
    w1 = torch.randn((n1, n2), requires_grad=True)
    b1 = torch.zeros((n1, 1), requires_grad=True)

    return w1, b1
