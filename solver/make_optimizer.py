import torch

def _generate_optimizer_class_with_freeze_layer(
        optimizer,
        *,
        freeze_epochs: int = 0,
):
    assert freeze_epochs > 0, "No layers need to be frozen or freeze iterations is 0"

    @torch.no_grad()
    def optimizer_wfl_step(self, cur_epoch, closure=None):
        if cur_epoch < freeze_epochs:
            param_ref = []
            grad_ref = []
            for group in self.param_groups:
                if group["freeze_status"] == "freeze":
                    for p in group["params"]:
                        if p.grad is not None:
                            param_ref.append(p)
                            grad_ref.append(p.grad)
                            p.grad = None

            optimizer.step(self, closure)
            for p, g in zip(param_ref, grad_ref):
                p.grad = g
        else:
            optimizer.step(self, closure)

    OptimizerWithFreezeLayer = type(
        optimizer.__name__ + "WithFreezeLayer",
        (optimizer,),
        {"step": optimizer_wfl_step},
    )
    return OptimizerWithFreezeLayer

def maybe_add_freeze_layer(
    cfg, 
    optimizer,
    ):
    if len(cfg.MODEL.FREEZE_LAYERS) == 0 or cfg.SOLVER.FREEZE_EPOCHS <= 0:
        return optimizer

    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    OptimizerWithFreezeLayer = _generate_optimizer_class_with_freeze_layer(
        optimizer_type,
        freeze_epochs=cfg.SOLVER.FREEZE_EPOCHS
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithFreezeLayer  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithFreezeLayer
    
def make_optimizer_fz(cfg, model, center_criterion):
    freeze_layers = cfg.MODEL.FREEZE_LAYERS
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        fz_status = 'normal'
        for fz_layer in freeze_layers:
            if fz_layer in key:
                fz_status = 'freeze'
                break
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "freeze_status": fz_status}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = maybe_add_freeze_layer(cfg, torch.optim.SGD)(params, momentum=cfg.SOLVER.MOMENTUM, )
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_gan(cfg, model, center_criterion):
    params = []
    params_bg_head = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        
        if '_bg' in key:
            params_bg_head += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        else:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        optimizer_bg = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_bg_head, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        optimizer_bg = torch.optim.AdamW(params_bg_head, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        optimizer_bg = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_bg_head)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center, optimizer_bg
