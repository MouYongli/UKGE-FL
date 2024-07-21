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

class KGTripleDataset(Dataset):
    def __init__(self, root: str=data_path, dataset: str='cn15k', split: str='train', num_neg_per_positive: int=10, topk=True, k=200, test_with_neg=False) -> None:
        super().__init__()
        self.root=root
        self.dataset = dataset
        self.split = split
        self.num_neg_per_positive = num_neg_per_positive
        self.topk = topk
        self.k = k
        self.test_with_neg = test_with_neg
        self.entity_id = pd.read_csv(os.path.join(self.root, dataset, 'entity_id.csv'))
        self.relation_id = pd.read_csv(os.path.join(self.root, dataset, 'relation_id.csv'))
        # concept vocab
        self.cons = []
        # rel vocab
        self.rels = []
        # transitive rels vocab
        self.index_cons = {}  # {string: index}
        self.index_rels = {}  # {string: index}
        # Load data into cons and index_cons
        for _, row in self.entity_id.iterrows():
            # Add entity to cons list
            self.cons.append(row['entity string'])
            # Add entity and id mapping to index_cons dictionary
            self.index_cons[row['entity string']] = row['id']
        # Load data into rels and index_rels
        for _, row in self.relation_id.iterrows():
            # Add entity to cons list
            self.rels.append(row['relation string'])
            # Add entity and id mapping to index_cons dictionary
            self.index_rels[row['relation string']] = row['id']
        all_triples_df = pd.read_csv(os.path.join(self.root, dataset, 'data.tsv'.format(split)), sep='\t', header=None)
        triples_df = pd.read_csv(os.path.join(self.root, self.dataset, '{}.tsv'.format(split)), sep='\t', header=None)

        # record all triples in the dataset, used for negative sampling in "train" set
        self.triples_record = set([])
        for h_, r_, t_, w in all_triples_df.to_numpy():
            h, r, t = int(h_), int(r_), int(t_)
            self.triples_record.add((h, r, t))

        # only used for "val" and "test" set
        if self.split != 'train':
            self.hr_ts_map = {}
            self.hr_all_ts_map = {}
            self.topk_hr_all_ts_map = {}
            for h_, r_, t_, w in triples_df.to_numpy():
                h, r, t = int(h_), int(r_), int(t_)
                if self.hr_ts_map.get(h) == None:
                    self.hr_ts_map[h] = {}
                    self.hr_all_ts_map[h] = {}
                if self.hr_ts_map[h].get(r) == None:
                    self.hr_ts_map[h][r] = {t: w}
                    self.hr_all_ts_map[h][r] = {t: w}
                else:
                    self.hr_ts_map[h][r][t] = w
                    self.hr_all_ts_map[h][r][t] = w
            for h_, r_, t_, w in all_triples_df.to_numpy():
                h, r, t = int(h_), int(r_), int(t_)
                if h in self.hr_all_ts_map and r in self.hr_all_ts_map[h]:
                    self.hr_all_ts_map[h][r][t] = w
            hr_num_t = {(h, r): len(self.hr_all_ts_map[h][r]) for h in self.hr_all_ts_map.keys() for r in self.hr_all_ts_map[h].keys()}
            topk_hr_num_t = sorted(hr_num_t.items(), key=lambda item: item[1], reverse=True)[:k]
            for (h, r), _ in topk_hr_num_t:
                if h not in self.topk_hr_all_ts_map:
                    self.topk_hr_all_ts_map[h] = {}
                if r not in self.topk_hr_all_ts_map[h]:
                    self.topk_hr_all_ts_map[h][r] = self.hr_all_ts_map[h][r]

        if self.split == 'test' and self.test_with_neg:
            triples_df = pd.read_csv(os.path.join(self.root, dataset, 'test_with_neg.tsv'), sep='\t', header=None)
            for h_, r_, t_, w in triples_df.to_numpy():
                h, r, t = int(h_), int(r_), int(t_)
                if self.hr_ts_map.get(h) == None:
                    self.hr_ts_map[h] = {}
                if self.hr_ts_map[h].get(r) == None:
                    self.hr_ts_map[h][r] = {t: w}
                else:
                    self.hr_ts_map[h][r][t] = w
        
        self.data = {
            'head_index': triples_df[0].to_numpy(),
            'rel_index': triples_df[1].to_numpy(),
            'tail_index': triples_df[2].to_numpy(),
            'score': triples_df[3].to_numpy(),
        }

    def get_hr_ts_map(self) -> dict:
        if self.split == 'train':
            raise ValueError("This method is only used for 'val' and 'test' set.")
        return self.hr_ts_map
    
    def get_hr_all_ts_map(self) -> dict:
        if self.split == 'train':
            raise ValueError("This method is only used for 'val' and 'test' set.")
        return self.hr_all_ts_map
    
    def get_topk_hr_all_ts_map(self) -> dict:
        if self.split == 'train' or self.topk == False:
            raise ValueError("This method is only used for 'val' and 'test' set when topk is True.")
        return self.topk_hr_all_ts_map

    def num_cons(self):
        '''Returns number of ontologies.

        This means all ontologies have index that 0 <= index < num_onto().
        '''
        return len(self.cons)

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return len(self.rels)

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_rels.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.rels[rel_index]

    def con_str2index(self, con_str):
        '''For ontology `con_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_cons.get(con_str)

    def con_index2str(self, con_index):
        '''For ontology `con_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.cons[con_index]
    
    def corrupt_pos(self, triple, pos):
        """
        :param triple: [h, r, t]
        :param pos: index position to replace (0 for h, 2 fot t)
        :return: [h', r, t] or [h, r, t']
        """
        hit = True
        res = None
        while hit:
            res = np.copy(triple)
            samp = np.random.randint(self.num_cons())
            while samp == triple[pos]:
                samp = np.random.randint(self.num_cons())
            res[pos] = samp
            if tuple(res) not in self.triples_record:
                hit = False
        return res
    
    # bernoulli negative sampling
    def corrupt(self, triple, neg_per_positive, tar=None):
        """
        :param triple: [h r t]
        :param tar: 't' or 'h'
        :return: np.array [[h,r,t1],[h,r,t2],...]
        """
        # print("array.shape:", res.shape)
        if tar == 't':
            position = 2
        elif tar == 'h':
            position = 0
        else:
            raise ValueError(f"value tar {tar} is not 't' or 'h'")
        res = [self.corrupt_pos(triple, position) for i in range(neg_per_positive)]
        return np.array(res)

    # bernoulli negative sampling on a batch
    def corrupt_batch(self, t_batch, neg_per_positive, tar=None):
        res = np.array([self.corrupt(triple, neg_per_positive, tar) for triple in t_batch])
        return res
    
    def __len__(self):
        return len(self.data['head_index'])
    
    def __getitem__(self, idx) -> Any:
        h = self.data['head_index'][idx]
        r = self.data['rel_index'][idx]
        t = self.data['tail_index'][idx]
        s = self.data['score'][idx]
        if self.split == 'train':
            nhrt = self.corrupt([h, r, t], self.num_neg_per_positive, tar='h')
            hrnt = self.corrupt([h, r, t], self.num_neg_per_positive, tar='t')
            return np.array([h, r, t]), np.array(s), nhrt, hrnt
        else:
            return np.array([h, r, t]), np.array(s)
        


# class DetKGTripleDataset(KGTripleDataset):
#     def __init__(self, root: str=data_path, dataset: str='cn15k', split: str='train', num_neg_per_positive: int=10, topk=True, k=200, deterministic=False, threshold=0.5):   
#         super().__init__(root=root, dataset=dataset, split=split, num_neg_per_positive=num_neg_per_positive)
#         self.deterministic = deterministic
#         self.threshold = threshold
#         data_triples_df = pd.read_csv(os.path.join(self.root, self.dataset, '{}.tsv'.format(self.split)), sep='\t', header=None)
#         all_data_triples_df = pd.read_csv(os.path.join(self.root, self.dataset, 'data.tsv'), sep='\t', header=None)

#         if self.deterministic and self.split == 'train':
#             data_triples_df = data_triples_df[data_triples_df[3] >= self.threshold]
#             all_data_triples_df = all_data_triples_df[all_data_triples_df[3] >= self.threshold]
            
#         self.data = {
#             'head_index': data_triples_df[0].to_numpy(),
#             'rel_index': data_triples_df[1].to_numpy(),
#             'tail_index': data_triples_df[2].to_numpy(),
#             'score': data_triples_df[3].to_numpy(),
#         }
        
#         self.triples_record = set([])

#         self.hrtw_map = {}
#         self.hr_all_tw_map = {}
#         for h_, r_, t_, w in data_triples_df.to_numpy():
#             h, r, t = int(h_), int(r_), int(t_)
#             if self.hrtw_map.get(h) == None:
#                 self.hrtw_map[h] = {}
#                 self.hr_all_tw_map[h] = {}
#             if self.hrtw_map[h].get(r) == None:
#                 self.hrtw_map[h][r] = {t: w}
#                 self.hr_all_tw_map[h][r] = {t: w}
#             else:
#                 self.hrtw_map[h][r][t] = w
#                 self.hr_all_tw_map[h][r][t] = w

#         for h_, r_, t_, w in all_data_triples_df.to_numpy():
#             h, r, t = int(h_), int(r_), int(t_)
#             self.triples_record.add((h, r, t))
#             if h in self.hr_all_tw_map and r in self.hr_all_tw_map[h]:
#                 self.hr_all_tw_map[h][r][t] = w



class KGPSLTripleDataset(Dataset):
    def __init__(self, root: str=data_path, dataset: str='cn15k'):
        self.root=root
        self.dataset = dataset
        self.entity_id = pd.read_csv(os.path.join(self.root, dataset, 'entity_id.csv'))
        self.relation_id = pd.read_csv(os.path.join(self.root, dataset, 'relation_id.csv'))
        psl_triples_df = pd.read_csv(os.path.join(self.root, dataset, 'softlogic.tsv'), sep='\t', header=None)
        self.data = {
            'head_index': psl_triples_df[0].to_numpy(),
            'rel_index': psl_triples_df[1].to_numpy(),
            'tail_index': psl_triples_df[2].to_numpy(),
            'score': psl_triples_df[3].to_numpy(),
        }
        
        # concept vocab
        self.cons = []
        # rel vocab
        self.rels = []
        # transitive rels vocab
        self.index_cons = {}  # {string: index}
        self.index_rels = {}  # {string: index}
        
        # Load data into cons and index_cons
        for _, row in self.entity_id.iterrows():
            # Add entity to cons list
            self.cons.append(row['entity string'])
            # Add entity and id mapping to index_cons dictionary
            self.index_cons[row['entity string']] = row['id']
        # Load data into rels and index_rels
        for _, row in self.relation_id.iterrows():
            # Add entity to cons list
            self.rels.append(row['relation string'])
            # Add entity and id mapping to index_cons dictionary
            self.index_rels[row['relation string']] = row['id']
    
    def __len__(self):
        return len(self.data['head_index'])
    
    def __getitem__(self, idx) -> Any:
        h = self.data['head_index'][idx]
        r = self.data['rel_index'][idx]
        t = self.data['tail_index'][idx]
        s = self.data['score'][idx]
        return np.array([h, r, t]), np.array(s)

if __name__ == "__main__":
    train_dataset = KGTripleDataset(split='train')
    val_dataset = KGTripleDataset(split='val')
    test_dataset = KGTripleDataset(split='test')
    psl_dataset = KGPSLTripleDataset()
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    print(len(psl_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    hrt, s, nhrt, hrnt = next(iter(train_dataloader))
    print(hrt.shape, s.shape)
    print(nhrt.shape)
    print(hrnt.shape)

    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    hrt, s = next(iter(val_dataloader))
    print(hrt.shape, s.shape)

    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    hrt, s = next(iter(test_dataloader))
    print(hrt.shape, s.shape)