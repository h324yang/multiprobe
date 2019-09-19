import torch


def kl_(p, q, eps=1e-7):
    p.div_(p.sum())
    q.div_(q.sum()).add_(eps)
    return p.mul_(p.div(q).add_(eps).log2_()).sum()

def jsd_(p, q):
    m = (p + q).div_(2)
    return kl_(p, m).add_(kl_(q, m)).div_(2).sqrt_()
