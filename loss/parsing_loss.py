import torch
import torch.nn.functional as F

def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2, weight=None):
    num_classes = pred_class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    
    if weight is not None:
        loss = (loss * weight).sum() / weight.sum().clamp(min=1)
    else:
        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
        loss = loss.sum() / non_zero_cnt

    return loss

def calculate_mask_loss(pred, targets, label_smoothing=0.0, weight=None):
    # weight: (B)
    # targets: (B, H, W)
    if weight is not None:
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand_as(targets)
        weight = weight.flatten()
    targets = targets.flatten()  # [b*h*w]
    pred = pred.flatten(0,1)  # [b*h*w, C]
    loss = cross_entropy_loss(pred, targets, eps=label_smoothing, weight=weight)
    return loss