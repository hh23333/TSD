import torch.nn.functional as F
import torch


def diverse_loss(feats):
    B, P, _ = feats.shape
    f_part_normed = F.normalize(feats, dim=-1)
    p_sim = torch.matmul(f_part_normed, f_part_normed.transpose(-2, -1)).relu()  # (B, P, P)
    loss_divs = p_sim.view(B, -1)[:, :-1].view(B, P-1, P+1)[:, :, 1:].mean()
    return loss_divs

def diverse_loss_w(feats, part_type=None):
    B, P, _ = feats.shape
    f_part_normed = F.normalize(feats, dim=-1)
    p_sim = torch.matmul(f_part_normed, f_part_normed.transpose(-2, -1)).relu()  # (B, P, P)
    loss_divs = p_sim.view(B, -1)[:, :-1].view(B, P-1, P+1)[:, :, 1:]
    if part_type is not None:
        if part_type == 'eight':
            part_index = torch.Tensor([0,1,1,2,3,3,4,4]).to(feats.device).unsqueeze(1)
            part_matrix = 1 - (part_index==part_index.t()).to(feats.dtype)
            part_matrix = part_matrix.unsqueeze(0).repeat(B, 1, 1)
            part_matrix = part_matrix.view(B, -1)[:, :-1].view(B, P-1, P+1)[:, :, 1:]
            loss_divs = (part_matrix * loss_divs).sum(-1)/part_matrix.sum(-1)
        elif part_type == 'five':
            part_index = torch.Tensor([0,1,2,3,4]).to(feats.device).unsqueeze(1)
            part_matrix = 1 - (part_index==part_index.t()).to(feats.dtype)
            part_matrix = part_matrix.unsqueeze(0).repeat(B, 1, 1)
            part_matrix = part_matrix.view(B, -1)[:, :-1].view(B, P-1, P+1)[:, :, 1:]
            loss_divs = (part_matrix * loss_divs).sum(-1)/part_matrix.sum(-1)
        elif part_type == 'three':
            part_index = torch.Tensor([0,1,2]).to(feats.device).unsqueeze(1)
            part_matrix = 1 - (part_index==part_index.t()).to(feats.dtype)
            part_matrix = part_matrix.unsqueeze(0).repeat(B, 1, 1)
            part_matrix = part_matrix.view(B, -1)[:, :-1].view(B, P-1, P+1)[:, :, 1:]
            loss_divs = (part_matrix * loss_divs).sum(-1)/part_matrix.sum(-1)
        loss_divs = loss_divs.mean()
    else:
        loss_divs = loss_divs.mean()
    return loss_divs