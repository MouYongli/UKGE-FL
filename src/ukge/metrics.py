import os
import os.path as osp
from typing import Any, List, Tuple
import numpy as np
import pandas as pd


from ukge.datasets import KGTripleDataset
from ukge.models import KGEModel

import torch
from tqdm import tqdm

class IndexScore:
    """
    The score of a tail when h and r is given.
    It's used in the ranking task to facilitate comparison and sorting.
    Print w as 3 digit precision float.
    """
    def __init__(self, index, score):
        self.index = index
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        # return "(index: %d, w:%.3f)" % (self.index, self.score)
        return "(%d, %.3f)" % (self.index, self.score)

    def __str__(self):
        return "(index: %d, w:%.3f)" % (self.index, self.score)


class Evaluator(object):
    def __init__(self, dataloader: KGTripleDataset, model: KGEModel, batch_size: int = 1024, device = None, topk=200):
        self.dataloader = dataloader
        self.model = model
        self.hrtw_map = self.dataloader.dataset.get_hrtw_map() # hrtw_map: {h: {r: {t: w}}}，只对于选择的dataset
        self.hr_all_tw_map = self.dataloader.dataset.get_hr_all_tw_map()
        self.num_cons = self.dataloader.dataset.num_cons()

        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hr_scores_map = {h:{r:[]} for h in self.hrtw_map.keys() for r in self.hrtw_map[h].keys()}

        # Filter out top 200 h,r based on the number of t
        hr_num_t = {(h, r): len(self.hr_all_tw_map[h][r]) for h in self.hr_all_tw_map.keys() for r in self.hr_all_tw_map[h].keys()}
        sorted_hr_num_t = sorted(hr_num_t.items(), key=lambda item: item[1], reverse=True)
        top_200_hr_num_t = sorted_hr_num_t[:topk]
        self.topk_hr_all_tw_map = {(h, r): self.hr_all_tw_map[h][r] for (h, r), _ in top_200_hr_num_t}

    def get_hr_scores(self, h: int, r: int) -> List[float]:
        """
        Get the scores of all tail entities for a given head and relation.
        """
        scores = []
        for t in range(0, self.num_cons, self.batch_size):
            batch_t = torch.arange(t, min(t + self.batch_size, self.num_cons)).to(self.device)
            batch_h = torch.full_like(batch_t, h).to(self.device)
            batch_r = torch.full_like(batch_t, r).to(self.device)
            batch_scores = self.model.get_confidence_score(batch_h, batch_r, batch_t)
            scores.extend(batch_scores.detach().cpu().numpy().tolist())
        return scores
    
    def update_hr_scores_map(self) -> dict:
        print("Updating hr_scores_map...")
        for i, h in enumerate(tqdm(self.hrtw_map.keys())):
            for r in self.hrtw_map[h].keys():
                self.hr_scores_map[h][r] = self.get_hr_scores(h, r) # list of scores for all tail entities（但没有显示对应的tail index

    def get_f1(self) ->  Tuple[Tuple[float], Tuple[float], Tuple[float]]:
        """
        Calculate the F1 score of the given list of (h, r, t, w, w_hat) tuples.
        """
        w = np.array([self.hrtw_map[h][r][t] for h in self.hrtw_map.keys() for r in self.hrtw_map[h].keys() for t in self.hrtw_map[h][r].keys()])
        w_hat = np.array([self.hr_scores_map[h][r][t]for h in self.hrtw_map.keys() for r in self.hrtw_map[h].keys() for t in self.hrtw_map[h][r].keys()])
        precisions, recalls, f1s = [], [], []
        for i in np.arange(0, 1, 0.05):
            w_t = (w > i).astype(int)
            p_t, r_t, f1_t = [], [], []
            for j in np.arange(0, 1, 0.05):
                w_hat_t = (w_hat > j).astype(int)
                tp = np.sum(w_t * w_hat_t)
                fp = np.sum((1 - w_t) * w_hat_t)
                fn = np.sum(w_t * (1 - w_hat_t))
                p = tp / (tp + fp) if tp + fp > 0 else 0
                r = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * p * r / (p + r) if p + r > 0 else 0
                p_t.append(p)
                r_t.append(r)
                f1_t.append(f1)
            precisions.append(p_t)
            recalls.append(r_t)
            f1s.append(f1_t)
        return precisions, recalls, f1s

    def get_mse(self) -> float:
        """
        Calculate the mean squared error of the given list of (h, r, t, w, w') tuples.
        """
        mse = 0.0
        count = 0
        for h in self.hrtw_map.keys():
            for r in self.hrtw_map[h].keys():
                for t in self.hrtw_map[h][r].keys():
                    w = self.hrtw_map[h][r][t]
                    w_hat = self.hr_scores_map[h][r][t]
                    mse += (w - w_hat) ** 2
                    count += 1
                mse += (w - w_hat) ** 2
        return mse / count
    
    def get_mae(self) -> float:
        """
        Calculate the mean absolute error of the given list of (h, r, t, w, w') tuples.
        """
        mae = 0.0
        count = 0
        for h in self.hrtw_map.keys():
            for r in self.hrtw_map[h].keys():
                for t in self.hrtw_map[h][r].keys():
                    w = self.hrtw_map[h][r][t]
                    w_hat = self.hr_scores_map[h][r][t]
                    mae += abs(w - w_hat)
                    count += 1
                mae += abs(w - w_hat)
        return mae / count

    def ndcg(self, h: int, r: int, tw_truth: List[IndexScore]) -> Tuple[float, float]:
        """
        Calculate the nDCG of the given list of (t, w) tuples.
        """
        ts = [tw.index for tw in tw_truth]  
        scores_array = np.array(self.hr_scores_map[h][r]) #hr_scores_map是list of scores for all tail entities（但没有显示对应的tail index
        scores_rank_array = scores_array.argsort()[::-1].argsort() + 1 #计算scores_array的排名（返回的是每个array对应的排名
        ranks = np.array([scores_rank_array[i] for i in ts]) #是不是默认scores array是按照index排序的（？
        #这里的
    
        # linear gain
        gains = np.array([tw.score for tw in tw_truth])
        discounts = np.log2(ranks + 1)
        discounted_gains = gains / discounts
        dcg = np.sum(discounted_gains)  # discounted cumulative gain
        # normalize
        max_possible_dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain
        # exponential gain
        exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
        exp_discounted_gains = exp_gains / discounts
        exp_dcg = np.sum(exp_discounted_gains)
        # normalize
        exp_max_possible_dcg = np.sum(
            exp_gains / np.log2(np.arange(len(exp_gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

        return ndcg, exp_ndcg

    def get_mean_ndcg(self):
        """
        Calculate the mean NDCG of the given list of (h, r, t, w, w') tuples.
        """
        ndcg_sum = 0  # nDCG with linear gain
        exp_ndcg_sum = 0  # nDCG with exponential gain
        count = 0
        for h in self.hr_all_tw_map.keys():
            for r in self.hr_all_tw_map[h].keys():
                tw_dict = self.hr_all_tw_map[h][r]
                tw_truth = [IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ndcg, exp_ndcg = self.ndcg(h, r, tw_truth)  # nDCG with linear gain and exponential gain
                ndcg_sum += ndcg
                exp_ndcg_sum += exp_ndcg
                count += 1
        return ndcg_sum / count, exp_ndcg_sum / count

    def get_mean_ndcg_topk(self):
        """
        Calculate the mean NDCG of the given list of (h, r, t, w, w') tuples.
        """
        ndcg_sum = 0  # nDCG with linear gain
        exp_ndcg_sum = 0  # nDCG with exponential gain
        count = 0
        for (h, r) in self.topk_hr_all_tw_map.keys():
            tw_dict = self.topk_hr_all_tw_map[h, r]
            tw_truth = [IndexScore(t, w) for t, w in tw_dict.items()]
            tw_truth.sort(reverse=True)  # descending on w
            ndcg, exp_ndcg = self.ndcg(h, r, tw_truth)  # nDCG with linear gain and exponential gain
            ndcg_sum += ndcg
            exp_ndcg_sum += exp_ndcg
            count += 1
        return ndcg_sum / count, exp_ndcg_sum / count

