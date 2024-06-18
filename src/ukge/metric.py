import os
import os.path as osp
from typing import Any, List, Tuple
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

here = osp.dirname(osp.abspath(__file__))
data_path = osp.join(here, '../..', 'data')

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
    def __init__(self, root: str=data_path, dataset: str='cn15k'):
        self.root=root
        self.dataset=dataset
        self.hr_map = {}
        all_data_triples_df = pd.read_csv(os.path.join(self.root, dataset, 'data.tsv'), sep='\t', header=None)
        for h_, r_, t_, w in all_data_triples_df.to_numpy():  # only training data
            h, r, t = int(h_), int(r_), int(t_)
            if self.hr_map.get(h) == None:
                self.hr_map[h] = {}
            if self.hr_map[h].get(r) == None:
                self.hr_map[h][r] = {t: w}
            else:
                self.hr_map[h][r][t] = w
    
if __name__ == "__main__":
    eval = Evaluator(dataset='cn15k')
    print(eval.hr_map)
