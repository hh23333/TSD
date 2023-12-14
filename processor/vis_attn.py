import cv2
import numpy as np
import os
import math

GRID_SPACING_V = 100
GRID_SPACING_H = 100
QUERY_EXTRA_SPACING = 30
TOP_MARGIN = 350
LEFT_MARGIN = 150
RIGHT_MARGIN = 500
BOTTOM_MARGIN = 300
ROW_BACKGROUND_LEFT_MARGIN = 75
ROW_BACKGROUND_RIGHT_MARGIN = 75
LEFT_TEXT_OFFSET = 10
BW = 12  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (255, 255, 0)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 0, 0)
TEXT_LINE_TYPE = cv2.LINE_AA
WIDTH = 192
HEIGHT = 256

import torch
import torch.nn.functional as F

class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        # model_type = model_dict['type']
        self.output_name = model_dict['output_name']  # ['global', 'foreground', 'concat']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        
        target_layer = self.model_arch.base.blocks[0].mlp.fc2

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def forward(self, input, class_idx=None, retain_graph=False, **kwargs):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        # targets = input["targets"]

        model_output, _ = self.model_arch(input, **kwargs)
        # cls_loss = model_output['loss_cls_'+self.output_name]
        # logits = model_output[self.output_name]['pred_class_logits']
        # score = logits[range(logits.size(0)), targets]
        # gd_targets = score.sum()
        gd_targets = model_output.sum()
        # if class_idx is None:
        #     score = logit[range(logit.size(0)), targets]  # TODO
        # else:
        #     score = logit[:, class_idx].squeeze()
        feat_h, feat_w = 16, 8
        self.model_arch.zero_grad()
        gd_targets.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'][:, 1:, :].permute(0, 2, 1).reshape(b, -1, feat_h, feat_w)
        activations = self.activations['value'][:, 1:, :].permute(0, 2, 1).reshape(b, -1, feat_h, feat_w)
        b, k, _, _ = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        # saliency_map = (activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False, **kwargs):
        return self.forward(input, class_idx, retain_graph, **kwargs)
    
def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result

def mask_overlay(img, mask, clip=True, interpolation=cv2.INTER_NEAREST):
    width, height = img.shape[1], img.shape[0]
    mask = cv2.resize(mask, dsize=(width, height), interpolation=interpolation)
    if clip:
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    masked_img = cv2.addWeighted(img, 0.5, mask_color.astype(img.dtype), 0.5, 0)
    return masked_img

def insert_img_into_grid(grid_img, img, row=0, col=0, text=None, fontScale=1.0, thickness=1, color=TEXT_COLOR):
    extra_spacing_h = QUERY_EXTRA_SPACING if row > 0 else 0
    extra_spacing_w = QUERY_EXTRA_SPACING if col > 0 else 0
    width, height = img.shape[1], img.shape[0]
    hs = row * height + (row + 1) * GRID_SPACING_V + extra_spacing_h + TOP_MARGIN
    he = (row + 1) * height + (row + 1) * GRID_SPACING_V + extra_spacing_h + TOP_MARGIN
    ws = col * width + (col + 1) * GRID_SPACING_H + extra_spacing_w + LEFT_MARGIN
    we = (col + 1) * width + (col + 1) * GRID_SPACING_H + extra_spacing_w + LEFT_MARGIN
    grid_img[hs:he, ws:we, :] = img
    if text is not None:
        text_lines = text.split('\n')
        text_line_height = cv2.getTextSize(text_lines[0], TEXT_FONT, fontScale, thickness)[0][1]
        textX = ws
        textY = he + 20 + text_line_height
        for i, text_line in enumerate(text_lines):
            pos = (textX, textY + (text_line_height + 20) * i)
            if i > 0:
                cv2.putText(grid_img, text_line, pos, TEXT_FONT, fontScale=fontScale, color=color, thickness=thickness,
                                lineType=TEXT_LINE_TYPE)
            else:
                cv2.putText(grid_img, text_line, pos, TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                                lineType=TEXT_LINE_TYPE)
                
def visualize_attn(attn_maps, img_path=None, num_attr=1, vis_dir='', f_h=16, f_w=8, 
                   right_bool=None, pred_bool=None, prefix='', score=None):
    # attn_maps: (num_layer, batch_size, part_num, L)
    num_cols = num_attr + 1
    num_rows = len(attn_maps)
    batch_size = attn_maps[0].size(0)

    for id in range(batch_size):
        grid_img = 255 * np.ones(
            (
                num_rows * HEIGHT + (num_rows + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + BOTTOM_MARGIN,
                num_cols * WIDTH + (num_cols + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN + RIGHT_MARGIN,
                3
            ),
            dtype=np.uint8
        )
        img_name = img_path[id].split('/')[-1]
        output_path = os.path.join(vis_dir, prefix + img_name)
        for col in range(0, num_cols):
            img2insert = cv2.imread(img_path[id])
            img2insert = cv2.resize(img2insert, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            if col > 0:
                for row in range(num_rows):
                    attn_maps_i = attn_maps[row][id][col-1][4:].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[row][id].mean(0)[col-1][1:].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[id].max(0)[0][col-1].view(f_h, f_w).cpu().numpy()
                    img2insert1 = mask_overlay(img2insert, attn_maps_i, clip=False, interpolation=cv2.INTER_CUBIC)
                    if row < num_rows - 1:
                        insert_img_into_grid(grid_img, img2insert1, col=col, row=row)
                    else:
                        if score is None:
                            text_str = f'{attn_maps_i.max():.4f}'
                        else:
                            text_str = f'{score[id, col-1].item():.4f}'
                    #     text_str = ATTR_NAME[col-1] + '\n' + str(pred_bool[id, col-1].item())
                    #     right_or_not = right_bool[id, col-1].item()
                    #     color = GREEN if right_or_not else RED
                        insert_img_into_grid(grid_img, img2insert1, col=col, row=row, text=text_str)
            else:
                insert_img_into_grid(grid_img, img2insert, col=col)
        
        cv2.imwrite(output_path, grid_img)

def visualize_attn_max(attn_maps, img_path=None, num_attr=1, vis_dir='', f_h=16, f_w=8, 
                   right_bool=None, pred_bool=None, prefix='', score=None):
    # attn_maps: (num_layer, batch_size, part_num, L)
    num_cols = num_attr
    num_rows = len(attn_maps)
    batch_size = attn_maps[0].size(0)

    attn_parts = []
    for attn_map in attn_maps:
        attn_max = attn_map[:, 1:4, 4:].max(dim=1, keepdim=True)[0].expand_as(attn_map[:, 1:4, 4:])
        attn_part = torch.where(attn_map[:, 1:4, 4:]==attn_max, attn_map[:, 1:4, 4:], attn_map.new_zeros(1))
        attn_parts.append(attn_part)

    for id in range(batch_size):
        grid_img = 255 * np.ones(
            (
                num_rows * HEIGHT + (num_rows + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + BOTTOM_MARGIN,
                num_cols * WIDTH + (num_cols + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN + RIGHT_MARGIN,
                3
            ),
            dtype=np.uint8
        )
        img_name = img_path[id].split('/')[-1]
        output_path = os.path.join(vis_dir, prefix + img_name)
        for col in range(0, num_cols):
            img2insert = cv2.imread(img_path[id])
            img2insert = cv2.resize(img2insert, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            if col > 0:
                for row in range(num_rows):
                    attn_maps_i = attn_parts[row][id][col-1].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[row][id].mean(0)[col-1][1:].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[id].max(0)[0][col-1].view(f_h, f_w).cpu().numpy()
                    img2insert1 = mask_overlay(img2insert, attn_maps_i, clip=False, interpolation=cv2.INTER_CUBIC)
                    if row < num_rows - 1:
                        insert_img_into_grid(grid_img, img2insert1, col=col, row=row)
                    else:
                        if score is None:
                            text_str = f'{attn_maps_i.max():.4f}'
                        else:
                            text_str = f'{score[id, col-1].item():.4f}'
                    #     text_str = ATTR_NAME[col-1] + '\n' + str(pred_bool[id, col-1].item())
                    #     right_or_not = right_bool[id, col-1].item()
                    #     color = GREEN if right_or_not else RED
                        insert_img_into_grid(grid_img, img2insert1, col=col, row=row, text=text_str)
            else:
                insert_img_into_grid(grid_img, img2insert, col=col)
        
        cv2.imwrite(output_path, grid_img)

def visualize_attn_len(images, attn_maps, batched_inputs=None, num_attr=1, vis_dir='', f_h=16, f_w=8, 
                   right_bool=None, pred_bool=None, prefix='', score=None, max_len=10):
    # attn_maps: (num_layer, batch_size, part_num, L)
    num_cols = num_attr + 1
    num_rows = len(attn_maps)  # layer num
    num_lines = math.ceil(num_attr/max_len) * num_rows
    num_lines_per_rows = math.ceil(num_attr/max_len)
    batch_size = images.size(0)

    for id in range(batch_size):
        grid_img = 255 * np.ones(
            (
                num_lines * HEIGHT + (num_lines + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + BOTTOM_MARGIN,
                max_len * WIDTH + (max_len + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN + RIGHT_MARGIN,
                3
            ),
            dtype=np.uint8
        )
        img_name = batched_inputs['img_paths'][id].split('/')[-1]
        output_path = os.path.join(vis_dir, prefix + img_name)
        for col in range(0, num_cols):
            img2insert_ori = images[id].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            r, g, b = cv2.split(img2insert_ori)
            img2insert_ori = cv2.merge([b, g, r])
            if col > 0:
                for row in range(num_rows):
                    attn_maps_i = attn_maps[row][id].mean(0)[col-1].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[id].max(0)[0][col-1].view(f_h, f_w).cpu().numpy()
                    img2insert = mask_overlay(img2insert_ori, attn_maps_i, clip=False, interpolation=cv2.INTER_CUBIC)
                    cur_row = row * num_lines_per_rows + (col-1)//max_len
                    cur_col = (col-1)%max_len + 1
                    if row < num_rows - 1:
                        insert_img_into_grid(grid_img, img2insert, col=cur_col, row=cur_row)
                    else:
                        if score is None:
                            text_str = f'{attn_maps_i.max():.4f}'
                        else:
                            text_str = f'{score[id, col-1].item():.4f}'
                    #     text_str = ATTR_NAME[col-1] + '\n' + str(pred_bool[id, col-1].item())
                    #     right_or_not = right_bool[id, col-1].item()
                    #     color = GREEN if right_or_not else RED
                        insert_img_into_grid(grid_img, img2insert, col=cur_col, row=cur_row, text=text_str)
            else:
                insert_img_into_grid(grid_img, img2insert_ori, col=col)
        
        cv2.imwrite(output_path, grid_img)

def visualize_attn_multihead(images, attn_maps, batched_inputs=None, num_attr=1, vis_dir='', f_h=16, f_w=8, 
                   right_bool=None, pred_bool=None, prefix='', score=None, max_len=10):
    # attn_maps: (num_layer, batch_size, part_num, L)
    num_cols = num_attr + 1
    num_rows = len(attn_maps)  # layer num
    num_lines = math.ceil(num_attr/max_len) * num_rows
    num_lines_per_rows = math.ceil(num_attr/max_len)
    batch_size = images.size(0)

    for id in range(batch_size):
        grid_img = 255 * np.ones(
            (
                num_lines * HEIGHT + (num_lines + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + BOTTOM_MARGIN,
                max_len * WIDTH + (max_len + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN + RIGHT_MARGIN,
                3
            ),
            dtype=np.uint8
        )
        img_name = batched_inputs['img_paths'][id].split('/')[-1]
        output_path = os.path.join(vis_dir, prefix + img_name)
        for col in range(0, num_cols):
            img2insert_ori = images[id].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            r, g, b = cv2.split(img2insert_ori)
            img2insert_ori = cv2.merge([b, g, r])
            if col > 0:
                for row in range(num_rows):
                    attn_maps_i = attn_maps[row][id][col-1][0].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[id].max(0)[0][col-1].view(f_h, f_w).cpu().numpy()
                    img2insert = mask_overlay(img2insert_ori, attn_maps_i, clip=False, interpolation=cv2.INTER_CUBIC)
                    cur_row = row * num_lines_per_rows + (col-1)//max_len
                    cur_col = (col-1)%max_len + 1
                    if score is None:
                        text_str = f'{attn_maps_i.max():.4f}'
                    else:
                        text_str = f'{score[id, col-1].item():.4f}'
                    #     text_str = ATTR_NAME[col-1] + '\n' + str(pred_bool[id, col-1].item())
                    #     right_or_not = right_bool[id, col-1].item()
                    #     color = GREEN if right_or_not else RED
                    insert_img_into_grid(grid_img, img2insert, col=cur_col, row=cur_row, text=text_str)
            else:
                insert_img_into_grid(grid_img, img2insert_ori, col=col)
        
        cv2.imwrite(output_path, grid_img)

def vis_sim(sim_mat, save_path, prefix='heatmap'):
    if sim_mat.ndim==3:
        for i in range(sim_mat.shape[0]):
            arr = sim_mat[i]
            img = (arr * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path, f'{prefix}_{i}.jpg'), heatmap)

def rollout_attn(att_mat, norm=False, start_layer=1):
    att_mat = torch.stack(att_mat)  # (N_l, B, N_h, H*W+1, H*W+1)
    if not norm:
        att_mat = torch.mean(att_mat, dim=2)  # (N_l, B, H*W+1, H*W+1)
    else:
        att_mat = att_mat.norm(dim=2, p=2)  # (N_l, B, H*W+1, H*W+1)
    residual_att = torch.eye(att_mat.size(2)).to(att_mat.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_att = []
    joint_att.append(aug_att_mat[0])

    for n in range(start_layer, aug_att_mat.size(0)):
        joint_att.append(torch.matmul(aug_att_mat[n], joint_att[n-1]))

    return joint_att

def mean_attn(att_mat, norm=False, start_layer=1):
    att_mat = torch.stack(att_mat)  # (N_l, B, N_h, H*W+1, H*W+1)
    if not norm:
        att_mat = torch.mean(att_mat, dim=2)  # (N_l, B, H*W+1, H*W+1)
    else:
        att_mat = att_mat.norm(dim=2, p=2)  # (N_l, B, H*W+1, H*W+1)

    joint_att = []
    # joint_att.append(att_mat[0])

    for n in range(att_mat.size(0)):
        joint_att.append(att_mat[n])

    return joint_att

def visualize_attn_decoder(attn_maps, img_path=None, num_attr=1, vis_dir='', f_h=16, f_w=8, 
                   right_bool=None, pred_bool=None, prefix='', score=None):
    # attn_maps: (num_layer, batch_size, part_num, L)
    num_cols = num_attr + 1
    num_rows = len(attn_maps)
    batch_size = attn_maps[0].size(0)

    for id in range(batch_size):
        grid_img = 255 * np.ones(
            (
                num_rows * HEIGHT + (num_rows + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + BOTTOM_MARGIN,
                num_cols * WIDTH + (num_cols + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN + RIGHT_MARGIN,
                3
            ),
            dtype=np.uint8
        )
        img_name = img_path[id].split('/')[-1]
        output_path = os.path.join(vis_dir, prefix + img_name)
        for col in range(0, num_cols):
            img2insert = cv2.imread(img_path[id])
            img2insert = cv2.resize(img2insert, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            if col > 0:
                for row in range(num_rows):
                    attn_maps_i = attn_maps[row][id][col-1].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[row][id].mean(0)[col-1][1:].view(f_h, f_w).cpu().numpy()
                    # attn_maps_i = attn_maps[id].max(0)[0][col-1].view(f_h, f_w).cpu().numpy()
                    img2insert1 = mask_overlay(img2insert, attn_maps_i, clip=False, interpolation=cv2.INTER_CUBIC)
                    if row < num_rows - 1:
                        insert_img_into_grid(grid_img, img2insert1, col=col, row=row)
                    else:
                        if score is None:
                            text_str = f'{attn_maps_i.max():.4f}'
                        else:
                            text_str = f'{score[id, col-1].item():.4f}'
                    #     text_str = ATTR_NAME[col-1] + '\n' + str(pred_bool[id, col-1].item())
                    #     right_or_not = right_bool[id, col-1].item()
                    #     color = GREEN if right_or_not else RED
                        insert_img_into_grid(grid_img, img2insert1, col=col, row=row, text=text_str)
            else:
                insert_img_into_grid(grid_img, img2insert, col=col)
        
        cv2.imwrite(output_path, grid_img)