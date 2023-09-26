import torch
import torch.nn as nn

def loss_function(q, k, queue):
    τ=0.05
    N = q.shape[0]
    C = q.shape[1]

    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),τ))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),τ)), dim=1)
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))