import torch
import numpy as np
import os
from utils.reranking import re_ranking

import matplotlib.pyplot as plt
import tqdm
from PIL import Image

from .metrics import euclidean_distance, euclidean_dist_3d
# from metrics_new import eval_func

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, g_olabel, ig_occls=None, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_valid_query = []
    num_valid_q = 0.  # number of valid query
    for i, q_idx in enumerate(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        #
        ol_remove = (g_olabel[order] == 3)
        if ig_occls is not None:
            for ig_occl in ig_occls:
                ol_remove = ol_remove | (g_olabel[order]==ig_occl)
        ol_remove = (g_pids[order] == q_pid) & ol_remove
        remove = remove | ol_remove
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        all_valid_query.append(i)
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, all_AP, all_valid_query


class Visualizer_new():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, ig_occls=None):
        super(Visualizer_new, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.ig_occls = ig_occls

    def reset(self):
        self.feats = []
        self.occ_lbs = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, occ_lb = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.occ_lbs.extend(occ_lb.cpu().numpy())

    def get_model_output(self, all_ap, valid_queries, dist, q_pids, g_pids, q_camids, g_camids, g_olabels, img_path_list):
        self.all_ap = all_ap
        self.valid_queries = valid_queries
        self.dist = dist
        self.sim = 1 - dist
        self.q_pids = q_pids
        self.g_pids = g_pids
        self.q_camids = q_camids
        self.g_camids = g_camids
        self.g_olabels = g_olabels
        self.q_img_path = img_path_list[:self.num_query]
        self.g_img_path = img_path_list[self.num_query:]

        self.indices = np.argsort(dist, axis=1)
        self.matches = (g_pids[self.indices] == q_pids[:, np.newaxis]).astype(np.int32)

        self.num_query = len(q_pids)

    def vis_rank(self, img_path_list, cfg, euclidean_dist=True):  # called after each epoch
        log_dir = cfg.OUTPUT_DIR
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        g_olabels = np.asarray(self.occ_lbs[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        elif euclidean_dist:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        else:
            distmat = 1 - torch.mm(qf, gf.t())
        cmc, mAP, APs, valid_queries = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, g_olabels, ig_occls=self.ig_occls)
        self.get_model_output(APs, valid_queries, distmat, q_pids, g_pids, q_camids, g_camids, g_olabels, img_path_list)

        log_dir = os.path.join(log_dir, 'vis_rank')
        os.makedirs(log_dir, exist_ok=True)
        print("Saving rank list result ...")
        self.vis_rank_list(APs, log_dir, vis_label=False, num_vis=len(self.valid_queries),
                           rank_sort='', max_rank=20)
        print("Finish saving rank list results!")
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
    
    def get_matched_result(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        ###
        ol_remove = (self.g_olabels[order] == 3)
        if self.ig_occls is not None:
            for ig_occl in self.ig_occls:
                ol_remove = ol_remove | (self.g_olabels[order]==ig_occl)
        ol_remove = (self.g_pids[order] == q_pid) & ol_remove
        remove = remove | ol_remove
        ###
        keep = np.invert(remove)
        cmc = self.matches[q_index][keep]
        sort_idx = order[keep]
        return cmc, sort_idx

    def vis_rank_list(self, aps, log_dir, vis_label, num_vis, rank_sort, max_rank):
        query_indices = np.argsort(aps)
        if rank_sort == 'descending': query_indices = query_indices[::-1]
        query_indices = query_indices[:int(num_vis)]
        if vis_label:
            fig, axes = plt.subplots(2, max_rank + 1, figsize=(3 * max_rank, 12))
        else:
            fig, axes = plt.subplots(1, max_rank + 1, figsize=(3 * max_rank, 6))
        for cnt, q_idx in enumerate(tqdm.tqdm(query_indices)):
            # all_imgs = []
            q_idx = self.valid_queries[q_idx]
            cmc, sort_idx = self.get_matched_result(q_idx)
            query_name = self.q_img_path[q_idx]
            query_img = Image.open(query_name)
            query_img = query_img.convert('RGB').resize((128, 256))
            cam_id = self.q_camids[q_idx]
            # all_imgs.append(query_img)
            query_img = np.asarray(query_img, dtype=np.uint8)
            plt.clf()
            ax = fig.add_subplot(1, max_rank + 1, 1)
            ax.imshow(query_img)
            ax.set_title('{:.4f}/cam{}'.format(self.all_ap[q_idx], cam_id))
            ax.axis("off")
            for i in range(max_rank):
                if vis_label:
                    ax = fig.add_subplot(2, max_rank + 1, i + 2)
                else:
                    ax = fig.add_subplot(1, max_rank + 1, i + 2)
                g_idx = sort_idx[i]
                gallery_name = self.g_img_path[g_idx]
                cam_id = self.g_camids[g_idx]
                gallery_img = Image.open(gallery_name).resize((128, 256))
                gallery_img = gallery_img.convert('RGB')
                # all_imgs.append(gallery_img)
                gallery_img = np.asarray(gallery_img, dtype=np.uint8)
                if cmc[i] == 1:
                    label = 'true'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1, edgecolor=(0, 1, 0),
                                               fill=False, linewidth=5))
                else:
                    label = 'false'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1,
                                               edgecolor=(1, 0, 0), fill=False, linewidth=5))
                ax.imshow(gallery_img)
                ax.set_title(f'{self.sim[q_idx, sort_idx[i]]:.3f}/{label}/cam{cam_id}')
                ax.axis("off")

            plt.tight_layout()
            filepath = os.path.join(log_dir, "{}".format(query_name.split('/')[-1]))
            fig.savefig(filepath)

class Visualizer_w_new():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, ig_occls=None):
        super(Visualizer_w_new, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.ig_occls = ig_occls

    def reset(self):
        self.feats = []
        self.occ_lbs = []
        self.weights = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, weight, pid, camid, occ_lb = output
        self.feats.append(feat.cpu())
        self.weights.append(weight.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.occ_lbs.extend(occ_lb.cpu().numpy())

    def get_model_output(self, all_ap, valid_queries, dist, q_pids, g_pids, q_camids, g_camids, g_olabels, img_path_list):
        self.all_ap = all_ap
        self.valid_queries = valid_queries
        self.dist = dist
        self.sim = 1 - dist
        self.q_pids = q_pids
        self.g_pids = g_pids
        self.q_camids = q_camids
        self.g_camids = g_camids
        self.g_olabels = g_olabels
        self.q_img_path = img_path_list[:self.num_query]
        self.g_img_path = img_path_list[self.num_query:]

        self.indices = np.argsort(dist, axis=1)
        self.matches = (g_pids[self.indices] == q_pids[:, np.newaxis]).astype(np.int32)

        self.num_query = len(q_pids)

    def vis_rank(self, img_path_list, cfg, euclidean_dist=True, prefix=''):  # called after each epoch
        log_dir = cfg.OUTPUT_DIR
        feats = torch.cat(self.feats, dim=0)
        weights = torch.cat(self.weights, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=2, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        qw = weights[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        gw = weights[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        g_olabels = np.asarray(self.occ_lbs[self.num_query:])

        qf = qf.permute(1, 0, 2)
        gf = gf.permute(1, 0, 2)
        distmat = euclidean_dist_3d(qf, gf, qw, gw)

        cmc, mAP, APs, valid_queries = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, g_olabels, ig_occls=self.ig_occls)
        self.get_model_output(APs, valid_queries, distmat, q_pids, g_pids, q_camids, g_camids, g_olabels, img_path_list)

        log_dir = os.path.join(log_dir, prefix+'vis_rank')
        os.makedirs(log_dir, exist_ok=True)
        print("Saving rank list result ...")
        self.vis_rank_list(APs, log_dir, vis_label=False, num_vis=len(self.valid_queries),
                           rank_sort='', max_rank=20)
        print("Finish saving rank list results!")
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
    
    def get_matched_result(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        ###
        ol_remove = (self.g_olabels[order] == 3)
        if self.ig_occls is not None:
            for ig_occl in self.ig_occls:
                ol_remove = ol_remove | (self.g_olabels[order]==ig_occl)
        ol_remove = (self.g_pids[order] == q_pid) & ol_remove
        remove = remove | ol_remove
        ###
        keep = np.invert(remove)
        cmc = self.matches[q_index][keep]
        sort_idx = order[keep]
        return cmc, sort_idx

    def vis_rank_list(self, aps, log_dir, vis_label, num_vis, rank_sort, max_rank):
        query_indices = np.argsort(aps)
        if rank_sort == 'descending': query_indices = query_indices[::-1]
        query_indices = query_indices[:int(num_vis)]
        if vis_label:
            fig, axes = plt.subplots(2, max_rank + 1, figsize=(3 * max_rank, 12))
        else:
            fig, axes = plt.subplots(1, max_rank + 1, figsize=(3 * max_rank, 6))
        for cnt, q_idx in enumerate(tqdm.tqdm(query_indices)):
            # all_imgs = []
            ap = aps[q_idx]
            q_idx = self.valid_queries[q_idx]
            cmc, sort_idx = self.get_matched_result(q_idx)
            query_name = self.q_img_path[q_idx]
            query_img = Image.open(query_name)
            query_img = query_img.convert('RGB').resize((128, 256))
            cam_id = self.q_camids[q_idx]
            # all_imgs.append(query_img)
            query_img = np.asarray(query_img, dtype=np.uint8)
            plt.clf()
            ax = fig.add_subplot(1, max_rank + 1, 1)
            ax.imshow(query_img)
            ax.set_title('{:.4f}/cam{}'.format(ap, cam_id))
            ax.axis("off")
            for i in range(max_rank):
                if vis_label:
                    ax = fig.add_subplot(2, max_rank + 1, i + 2)
                else:
                    ax = fig.add_subplot(1, max_rank + 1, i + 2)
                g_idx = sort_idx[i]
                gallery_name = self.g_img_path[g_idx]
                cam_id = self.g_camids[g_idx]
                g_pid = self.g_pids[g_idx]
                gallery_img = Image.open(gallery_name).resize((128, 256))
                gallery_img = gallery_img.convert('RGB')
                # all_imgs.append(gallery_img)
                gallery_img = np.asarray(gallery_img, dtype=np.uint8)
                if cmc[i] == 1:
                    label = 'true'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1, edgecolor=(0, 1, 0),
                                               fill=False, linewidth=5))
                else:
                    label = 'false'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1,
                                               edgecolor=(1, 0, 0), fill=False, linewidth=5))
                ax.imshow(gallery_img)
                ax.set_title(f'{self.sim[q_idx, sort_idx[i]]:.3f}/{g_pid}/cam{cam_id}')
                ax.axis("off")

            plt.tight_layout()
            filepath = os.path.join(log_dir, "{}".format(query_name.split('/')[-1]))
            fig.savefig(filepath)